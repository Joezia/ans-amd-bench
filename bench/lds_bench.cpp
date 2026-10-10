#include "lds_format.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <cstdio>
using namespace ldsans;
#ifndef LDS_BUILD_FLAGS
#define LDS_BUILD_FLAGS "unknown"
#endif
static void check(hipError_t rc) {if(rc!=hipSuccess)throw std::runtime_error(hipGetErrorString(rc));}
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) {check(hipMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T)));}
  ~Device(){if(p)(void)hipFree(p);}
  void upload(const T* src,size_t n){if(n)check(hipMemcpy(p,src,n*sizeof(T),hipMemcpyHostToDevice));}
};

// Baseline deliberately uses independent scalar rANS lanes, not an optimized
// hipANS stream layout. Artifact v1 is incompatible with hipANS archives.
template<bool Verify,bool SharedTable,int Mode>
__global__ void ldsBaseline(const uint8_t* payload,const Stream* streams,
    const uint32_t* tableGlobal,const uint8_t* raw,uint8_t* output,uint32_t* error) {
  __shared__ uint32_t lookup[SharedTable ? TableSize : 1];
  __shared__ volatile uint8_t tile[8192];
  unsigned lane=threadIdx.x;
  if constexpr(SharedTable) {
    for(unsigned i=lane;i<TableSize;i+=64)lookup[i]=tableGlobal[i];
    __syncthreads();
  }
  uint32_t state=Lower, pos=0;
  Stream desc{};
  if constexpr(Mode==0) {desc=streams[uint64_t(blockIdx.x)*64+lane];state=desc.state;}
  const uint32_t* tablePtr=SharedTable ? lookup : tableGlobal;
  for(unsigned i=0;i<128;i++) {
    uint32_t symbol;
    if constexpr(Mode==0) {
      uint32_t slot=state&4095, entry=tablePtr[slot];symbol=entry&255;
      state=((entry>>20)+1)*(state>>12)+slot-((entry>>8)&4095);
      while(state<Lower) {
        // Bounds guard also retained in performance path for malformed input.
        if(pos>=desc.length) {if constexpr(Verify)atomicOr(error,1u);state=Lower;break;}
        state=(state<<8)|payload[desc.offset+pos++];
      }
    } else if constexpr(Mode==1) symbol=raw[uint64_t(blockIdx.x)*8192+i*64+lane];
    else symbol=(i*64+lane)&255;
    // HIP's generic volatile shared pointer can lower to FLAT stores. Force the
    // intended LDS instruction so the discard path is both retained and auditable.
    uint32_t address=uint32_t(reinterpret_cast<uintptr_t>(&tile[i*64+lane]));
    asm volatile("ds_write_b8 %0, %1" :: "v"(address), "v"(symbol) : "memory");
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __syncthreads(); // consumer boundary; verification reads actual LDS after this barrier
  if constexpr(Verify) {
    if constexpr(Mode==0)if(state!=Lower||pos!=desc.length)atomicOr(error,2u);
    for(unsigned i=lane;i<8192;i+=64)output[uint64_t(blockIdx.x)*8192+i]=tile[i];
  }
}

static std::string escape(const std::string& s) {
  std::string o;for(unsigned char c:s){if(c=='"'||c=='\\')o+='\\';if(c>=32)o+=c;}return o;
}
static std::string command(const std::string& cmd) {
  std::string s;FILE* f=popen(cmd.c_str(),"r");if(!f)return "unknown";
  char b[256];while(fgets(b,sizeof(b),f))s+=b;int rc=pclose(f);
  if(rc)return "unknown";while(!s.empty()&&(s.back()=='\n'||s.back()=='\r'))s.pop_back();return s;
}
static std::string quote(const std::string& s) {std::string o="'";for(char c:s)o+=c=='\''?"'\\''":std::string(1,c);return o+"'";}
int main(int argc,char**argv){try {
  std::string artifact,reference,mode="decode",tableMode="lds";
  int warmup=10,iters=50;bool skip=false;
  for(int i=1;i<argc;i++) {
    std::string arg=argv[i];
    if(arg=="--skip-verify"){skip=true;continue;}
    if(i+1>=argc)throw std::runtime_error("missing CLI value");std::string value=argv[++i];
    if(arg=="--artifact")artifact=value;else if(arg=="--reference")reference=value;
    else if(arg=="--mode")mode=value;else if(arg=="--table")tableMode=value;
    else if(arg=="--warmup"||arg=="--iters") {size_t end;int n=std::stoi(value,&end);if(end!=value.size())throw std::runtime_error("invalid iteration count");(arg=="--iters"?iters:warmup)=n;}
    else throw std::runtime_error("unknown option: "+arg);
  }
  if(artifact.empty()||reference.empty()||iters<1||warmup<0||
      (mode!="decode"&&mode!="raw"&&mode!="lds-control")||(tableMode!="lds"&&tableMode!="global"))
    throw std::runtime_error("usage: lds_bench --artifact file --reference weights.bin [--mode decode|raw|lds-control --table lds|global --warmup N --iters N]");
  auto a=load(artifact.c_str());auto lut=table(a.freq);
  auto raw=readBytes(reference.c_str());auto tiled=tileWeights(raw,a.h);
  int device;check(hipGetDevice(&device));hipDeviceProp_t prop;check(hipGetDeviceProperties(&prop,device));
  if(prop.warpSize!=64)throw std::runtime_error("artifact v1 requires wave64");
  if(uint64_t(a.h.tiles)>uint64_t(prop.maxGridSize[0]))throw std::runtime_error("grid exceeds device limits");
  Device<uint8_t> payload(a.payload.size()),rawDevice(tiled.size()),out(tiled.size());
  Device<Stream> streams(a.streams.size());Device<uint32_t> lutDevice(lut.size()),error(1);
  payload.upload(a.payload.data(),a.payload.size());streams.upload(a.streams.data(),a.streams.size());
  rawDevice.upload(tiled.data(),tiled.size());lutDevice.upload(lut.data(),lut.size());
  hipStream_t stream;check(hipStreamCreate(&stream));
  auto launch=[&](bool verify) {
    auto p=payload.p;auto s=streams.p;auto l=lutDevice.p;auto r=rawDevice.p;auto o=out.p;auto e=error.p;
    dim3 grid(a.h.tiles),block(64);
    if(mode=="decode") {
      if(tableMode=="lds") {
        if(verify)hipLaunchKernelGGL((ldsBaseline<true,true,0>),grid,block,0,stream,p,s,l,r,o,e);
        else hipLaunchKernelGGL((ldsBaseline<false,true,0>),grid,block,0,stream,p,s,l,r,o,e);
      }else{
        if(verify)hipLaunchKernelGGL((ldsBaseline<true,false,0>),grid,block,0,stream,p,s,l,r,o,e);
        else hipLaunchKernelGGL((ldsBaseline<false,false,0>),grid,block,0,stream,p,s,l,r,o,e);
      }
    }else if(mode=="raw") {
      if(verify)hipLaunchKernelGGL((ldsBaseline<true,false,1>),grid,block,0,stream,p,s,l,r,o,e);
      else hipLaunchKernelGGL((ldsBaseline<false,false,1>),grid,block,0,stream,p,s,l,r,o,e);
    }else hipLaunchKernelGGL((ldsBaseline<false,false,2>),grid,block,0,stream,p,s,l,r,o,e);
    check(hipGetLastError());
  };
  if(!skip&&mode!="lds-control") {
    check(hipMemset(error.p,0,4));launch(true);check(hipStreamSynchronize(stream));
    uint32_t errors;check(hipMemcpy(&errors,error.p,4,hipMemcpyDeviceToHost));
    std::vector<uint8_t> decoded(tiled.size());check(hipMemcpy(decoded.data(),out.p,decoded.size(),hipMemcpyDeviceToHost));
    if(errors||decoded!=tiled)throw std::runtime_error("actual LDS byte-exact roundtrip failed");
  }
  for(int i=0;i<warmup;i++)launch(false);check(hipStreamSynchronize(stream));
  hipEvent_t start,stop;check(hipEventCreate(&start));check(hipEventCreate(&stop));
  std::vector<double> samples;
  for(int i=0;i<iters;i++) {
    check(hipEventRecord(start,stream));launch(false);check(hipEventRecord(stop,stream));
    check(hipEventSynchronize(stop));float ms;check(hipEventElapsedTime(&ms,start,stop));samples.push_back(ms);
  }
  std::sort(samples.begin(),samples.end());auto pct=[&](double p){double v=p*(samples.size()-1);size_t lo=v;return samples[lo]+(samples[std::min(lo+1,samples.size()-1)]-samples[lo])*(v-lo);};
  double median=pct(.5);
  if(median<=0)throw std::runtime_error("invalid event latency");
  int runtime;check(hipRuntimeGetVersion(&runtime));
  const char* env=std::getenv("HIPANS_ROOT");std::string hipans=env?env:"../hipANS";
  auto commit=command("git -C "+quote(hipans)+" rev-parse HEAD 2>/dev/null");
  auto dirty=command("git -C "+quote(hipans)+" status --porcelain 2>/dev/null");
  auto rocm=command("hipconfig --path 2>/dev/null");
  std::ifstream version(rocm+"/.info/version");std::getline(version,rocm);
  uint64_t metadata=64+1024+a.streams.size()*16;
  std::cout<<std::setprecision(10)<<"{\"schema\":\"lds-bench-v1\",\"mode\":\""<<mode<<"\",\"table_mode\":\""<<tableMode<<"\",\"verified_actual_LDS\":"<<(!skip&&mode!="lds-control"?"true":"false")
    <<",\"raw_bytes\":"<<a.h.raw_bytes<<",\"padded_decoded_bytes\":"<<tiled.size()<<",\"unique_weights\":"<<a.h.raw_bytes
    <<",\"weights_produced_per_launch\":"<<a.h.raw_bytes<<",\"padding_bytes\":"<<tiled.size()-a.h.raw_bytes
    <<",\"compressed_payload_bytes\":"<<a.h.payload_bytes<<",\"artifact_metadata_bytes\":"<<metadata
    <<",\"artifact_bytes\":"<<a.h.payload_bytes+metadata<<",\"ratio_codes\":"<<double(a.h.raw_bytes)/(a.h.payload_bytes+metadata)
    <<",\"decode_ms\":{\"median\":"<<median<<",\"p10\":"<<pct(.1)<<",\"p90\":"<<pct(.9)<<"}"
    <<",\"decoded_LDS_GBps\":"<<tiled.size()/median/1e6<<",\"valid_weight_GBps\":"<<a.h.raw_bytes/median/1e6
    <<",\"weights_per_second\":"<<a.h.raw_bytes/median*1e3
    <<",\"compressed_payload_GBps\":"<<(mode=="decode"?a.h.payload_bytes/median/1e6:0)
    <<",\"warmup\":"<<warmup<<",\"iters\":"<<iters
    <<",\"cache_policy\":\"warm repeated tensor; no cache flush\",\"sink\":\"volatile inline-assembly ds_write_b8; final wait/barrier; no checksum\""
    <<",\"provenance\":{\"gpu_name\":\""<<escape(prop.name)<<"\",\"gpu_arch\":\""<<escape(prop.gcnArchName)<<"\",\"ROCm_version\":\""<<escape(rocm)<<"\",\"hip_runtime_version\":"<<runtime
    <<",\"hipANS_git_commit\":\""<<escape(commit)<<"\",\"hipANS_dirty\":"<<(dirty=="unknown"?"null":dirty.empty()?"false":"true")
    <<",\"codec\":\"local scalar byte-rANS baseline; not hipANS archive\",\"build_flags\":\""<<escape(LDS_BUILD_FLAGS)<<"\"}"
    <<",\"launch\":{\"grid\":["<<a.h.tiles<<",1,1],\"block\":[64,1,1],\"streams_per_workgroup\":64,\"tile_rows\":64,\"tile_cols\":128,\"LDS_bytes\":"<<(mode=="decode"&&tableMode=="lds"?24576:8192)<<"}}\n";
  check(hipEventDestroy(start));check(hipEventDestroy(stop));check(hipStreamDestroy(stream));
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
