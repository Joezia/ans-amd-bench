// S1 baseline: call the CURRENT hipANS decoder helpers, never a replacement rANS.
#include "hipans/ans/GpuANSDecode.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef HIPANS_ROOT_BUILT
#define HIPANS_ROOT_BUILT "../hipANS"
#endif
#ifndef HIPANS_COMMIT_BUILT
#define HIPANS_COMMIT_BUILT "unknown"
#endif
#ifndef HIPANS_LDS_BUILD_FLAGS
#define HIPANS_LDS_BUILD_FLAGS "unknown"
#endif
#ifndef HIPANS_GPU_ARCH_BUILT
#define HIPANS_GPU_ARCH_BUILT "gfx942"
#endif
namespace ha = hipans;
static void check(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template<class T> struct Device {
  T* p=nullptr;
  explicit Device(size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p),std::max(size_t(1),n)*sizeof(T))); }
  ~Device() { if(p)(void)hipFree(p); }
};
static std::vector<uint8_t> read(const std::string& path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f)throw std::runtime_error("cannot open "+path);
  auto n=f.tellg();if(n<0)throw std::runtime_error("invalid file length");
  std::vector<uint8_t> b(static_cast<size_t>(n));f.seekg(0);
  f.read(reinterpret_cast<char*>(b.data()),b.size());if(!f)throw std::runtime_error("short file");return b;
}
// Same matrix-to-tile mapping as the real-weight extraction delivery, not codec logic.
static std::vector<uint8_t> tiles(const std::vector<uint8_t>& raw,uint32_t rows,uint32_t cols) {
  if(raw.size()!=uint64_t(rows)*cols)throw std::runtime_error("reference shape mismatch");
  uint64_t nr=(uint64_t(rows)+63)/64,nc=(uint64_t(cols)+127)/128;
  if(nr*nc*8192>uint64_t(INT32_MAX))throw std::runtime_error("archive exceeds hipANS single-input limit");
  std::vector<uint8_t> out(nr*nc*8192,0);
  for(uint64_t t=0;t<nr*nc;t++)for(uint32_t i=0;i<8192;i++) {
    uint64_t r=t/nc*64+i/128,c=t%nc*128+i%128;
    if(r<rows&&c<cols)out[t*8192+i]=raw[r*cols+c];
  }
  return out;
}
struct LDSWriter {
  volatile uint8_t* tile;
  bool active;
  __device__ void write(uint32_t offset,uint8_t sym) {
    if(active) {
      uint32_t address=uint32_t(reinterpret_cast<uintptr_t>(tile+offset));
      uint32_t value=sym;
      asm volatile("ds_write_b8 %0, %1" :: "v"(address),"v"(value) : "memory");
    }
  }
};

template<int Threads,bool Verify,bool Raw>
__global__ __launch_bounds__(Threads) void hipansLDS(
    const uint8_t* archives,uint64_t stride,const ha::TableT* table,
    const uint8_t* raw,uint8_t* output,uint32_t blocks,uint32_t replicas,
    uint64_t groupBegin,uint64_t groupEnd) {
  extern __shared__ __align__(16) uint8_t storage[];
  auto lookup=reinterpret_cast<ha::TableT*>(storage);
  volatile uint8_t* tile=storage+(Raw?0:4096);
  unsigned tid=threadIdx.x,warp=tid/64,lane=ha::getLaneId();
  constexpr unsigned Waves=Threads/64;
  if constexpr(!Raw) {
    // Preserve the original hipANS uint4 LUT staging; only loop bounds support
    // additional deployment sizes. Probability precision remains the default 10.
    auto dst=reinterpret_cast<uint4*>(lookup);
    auto src=reinterpret_cast<const uint4*>(table);
    for(unsigned j=tid;j<256;j+=Threads)dst[j]=src[j];
    __syncthreads();
  }
  const uint64_t total=uint64_t(blocks)*replicas;
  for(uint64_t group=groupBegin+blockIdx.x;group<groupEnd;group+=gridDim.x) {
    uint64_t task=group*Waves+warp;
    bool active=task<total;
    // All waves must execute equal 8192-byte paths because the CURRENT decoder
    // has workgroup barriers inside decodeOneWarp. Tail waves decode block 0,
    // discard it, and never contribute to the effective numerator.
    uint64_t safeTask=active?task:0;
    unsigned rep=safeTask/blocks,block=safeTask%blocks;
    LDSWriter writer{tile+warp*8192,active};
    if constexpr(Raw) {
      for(unsigned i=lane;i<8192;i+=64)writer.write(i,raw[safeTask*8192+i]);
    } else {
      auto header=reinterpret_cast<const ha::ANSCoalescedHeader*>(archives+uint64_t(rep)*stride);
      auto state=header->getWarpStates()[block].warpState[lane];
      auto words=header->getBlockWords(blocks)[block];
      auto data=header->getBlockDataStart(blocks)+words.y;
      // This is the actual upstream implementation, including its original
      // state recurrence, 16-bit refill, ballot/prefix and per-symbol barrier.
      ha::ANSDecodeWarpFullBlock<LDSWriter,10,ha::kDefaultBlockSize,false>::decode(
          lane,state,words.x&0xffff,data,writer,lookup,block);
    }
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
    if constexpr(Verify) {
      for(unsigned i=tid;i<Waves*8192;i+=Threads) {
        uint64_t owner=group*Waves+i/8192;
        if(owner<total)output[owner*8192+i%8192]=tile[i];
      }
      __syncthreads();
    }
  }
}
template<int T> static const void* kernel(bool verify,bool raw) {
  if(raw)return verify?reinterpret_cast<const void*>(hipansLDS<T,true,true>):reinterpret_cast<const void*>(hipansLDS<T,false,true>);
  return verify?reinterpret_cast<const void*>(hipansLDS<T,true,false>):reinterpret_cast<const void*>(hipansLDS<T,false,false>);
}
static const void* select(int t,bool verify,bool raw) {
  switch(t) {
#define CASE(T) case T:return kernel<T>(verify,raw)
    CASE(64);CASE(128);CASE(192);CASE(256);CASE(320);CASE(384);CASE(448);
#undef CASE
  }
  throw std::runtime_error("threads must be 64,128,192,256,320,384,448");
}
struct Options {
  std::string artifact,reference,mode="decode";
  uint32_t rows=0,cols=0,replicas=1,threads=128,gridMult=4,streams=1,warmup=5,iters=30;
  bool prepare=false,replace=false,skip=false;
};
static Options options(int argc,char**argv) {
  Options a;
  for(int i=1;i<argc;i++) {
    std::string k=argv[i];
    if(k=="--prepare"){a.prepare=true;continue;}
    if(k=="--replace"){a.replace=true;continue;}
    if(k=="--skip-verify"){a.skip=true;continue;}
    if(i+1>=argc)throw std::runtime_error("missing argument value");std::string v=argv[++i];
    if(k=="--artifact")a.artifact=v;else if(k=="--reference")a.reference=v;
    else if(k=="--mode")a.mode=v;
    else {
      size_t end;uint64_t n=std::stoull(v,&end);
      if(end!=v.size()||n>UINT32_MAX||v[0]=='-')throw std::runtime_error("invalid numeric argument");
      if(k=="--rows")a.rows=n;else if(k=="--cols")a.cols=n;
      else if(k=="--replicas")a.replicas=n;else if(k=="--threads")a.threads=n;
      else if(k=="--grid-mult")a.gridMult=n;else if(k=="--streams")a.streams=n;
      else if(k=="--warmup")a.warmup=n;else if(k=="--iters")a.iters=n;
      else throw std::runtime_error("unknown option "+k);
    }
  }
  if(a.artifact.empty()||a.reference.empty()||!a.rows||!a.cols||!a.replicas||a.replicas>256||
      !a.gridMult||a.gridMult>16||!a.streams||a.streams>32||!a.iters||
      (a.mode!="decode"&&a.mode!="raw")||(a.prepare&&(a.skip||a.mode!="decode"||a.replicas!=1)))
    throw std::runtime_error("invalid CLI configuration");
  select(a.threads,false,false);return a;
}
static ha::ANSCoalescedHeader validate(const std::vector<uint8_t>& b,size_t expected) {
  if(b.size()<32)throw std::runtime_error("short native archive");
  ha::ANSCoalescedHeader h;std::memcpy(&h,b.data(),32);
  uint32_t magic;std::memcpy(&magic,b.data(),4);
  if(magic!=((ha::kANSMagic<<16)|ha::kANSVersion)||h.getProbBits()!=10||
      h.getUseChecksum()||h.getTotalUncompressedWords()!=expected||
      h.getNumBlocks()!=expected/8192||h.getTotalCompressedSize()!=b.size())
    throw std::runtime_error("archive is not the expected native hipANS p10 full-tile artifact");
  uint16_t probs[256];std::memcpy(probs,b.data()+32,512);uint32_t sum=0;
  for(auto p:probs)sum+=p;if(sum!=1024)throw std::runtime_error("invalid probability sum");
  size_t descriptors=32+512+size_t(h.getNumBlocks())*sizeof(ha::ANSWarpState);
  for(uint32_t i=0;i<h.getNumBlocks();i++) {
    uint2 w;std::memcpy(&w,b.data()+descriptors+i*8,8);
    if((w.x>>16)!=8192||uint64_t(w.y)+(w.x&0xffff)>h.getTotalCompressedWords())
      throw std::runtime_error("invalid native block descriptor");
  }
  return h;
}
int main(int argc,char**argv) {try {
  auto a=options(argc,argv);auto raw=read(a.reference);auto tiled=tiles(raw,a.rows,a.cols);
  int dev;check(hipGetDevice(&dev));hipDeviceProp_t props;check(hipGetDeviceProperties(&props,dev));
  if(props.warpSize!=64||std::string(props.gcnArchName).rfind(HIPANS_GPU_ARCH_BUILT,0)!=0)
    throw std::runtime_error("GPU must match the compiled S1 wave64 target");
  hipStream_t control;check(hipStreamCreateWithFlags(&control,hipStreamNonBlocking));
  std::vector<uint8_t> archive;
  if(a.prepare) {
    if(std::ifstream(a.artifact)&&!a.replace)throw std::runtime_error("artifact exists; explicit --replace required");
    Device<uint8_t> input(tiled.size()),encoded(ha::getMaxCompressedSize(tiled.size()));Device<uint32_t> size(1);
    check(hipMemcpyAsync(input.p,tiled.data(),tiled.size(),hipMemcpyHostToDevice,control));
    auto stack=ha::makeStackMemory(1ull<<30);
    const void* in=input.p;void* out=encoded.p;uint32_t bytes=tiled.size();
    ha::ansEncodeBatchPointer(stack,ha::ANSCodecConfig(10,false),1,&in,&bytes,nullptr,&out,size.p,control);
    check(hipStreamSynchronize(control));uint32_t n;
    check(hipMemcpy(&n,size.p,4,hipMemcpyDeviceToHost));
    if(n>ha::getMaxCompressedSize(tiled.size()))throw std::runtime_error("encoded size exceeds allocation");
    archive.resize(n);check(hipMemcpy(archive.data(),encoded.p,n,hipMemcpyDeviceToHost));
  } else archive=read(a.artifact);
  auto h=validate(archive,tiled.size());uint32_t blocks=h.getNumBlocks();
  uint64_t stride=(archive.size()+31)&~uint64_t(31),totalBytes=uint64_t(tiled.size())*a.replicas;
  Device<uint8_t> archives(stride*a.replicas),rawDevice(a.mode=="raw"?totalBytes:0),output(a.skip?0:totalBytes);
  Device<ha::TableT> lookup(1024);
  for(uint32_t r=0;r<a.replicas;r++) {
    check(hipMemcpyAsync(archives.p+r*stride,archive.data(),archive.size(),hipMemcpyHostToDevice,control));
    if(a.mode=="raw")check(hipMemcpyAsync(rawDevice.p+uint64_t(r)*tiled.size(),tiled.data(),tiled.size(),hipMemcpyHostToDevice,control));
  }
  ha::BatchProviderStride provider(archives.p,stride);
  hipEvent_t begin,end;check(hipEventCreate(&begin));check(hipEventCreate(&end));
  check(hipEventRecord(begin,control));
  hipLaunchKernelGGL((ha::ansDecodeTable<ha::BatchProviderStride,512>),dim3(1),dim3(512),0,control,provider,10,lookup.p);
  check(hipGetLastError());check(hipEventRecord(end,control));check(hipEventSynchronize(end));
  float tableMs;check(hipEventElapsedTime(&tableMs,begin,end));
  bool rawMode=a.mode=="raw";size_t shared=(a.threads/64)*8192+(rawMode?0:4096);
  if(shared>props.sharedMemPerBlock)throw std::runtime_error("LDS allocation exceeds device limit");
  const void* timedKernel=select(a.threads,false,rawMode);int activeBlocks;
  check(hipOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocks,timedKernel,a.threads,shared));
  if(!activeBlocks)throw std::runtime_error("no resident workgroups");
  hipFuncAttributes attrs;check(hipFuncGetAttributes(&attrs,timedKernel));
  uint64_t groups=(uint64_t(blocks)*a.replicas+(a.threads/64)-1)/(a.threads/64);
  std::vector<hipStream_t> streams(a.streams);std::vector<hipEvent_t> finished(a.streams);
  for(unsigned s=0;s<a.streams;s++){check(hipStreamCreateWithFlags(&streams[s],hipStreamNonBlocking));check(hipEventCreateWithFlags(&finished[s],hipEventDisableTiming));}
  unsigned gridCap=activeBlocks*props.multiProcessorCount*a.gridMult;
  std::vector<unsigned> grids(a.streams);
  auto launch=[&](bool verify) {
    const void* f=select(a.threads,verify,rawMode);
    for(unsigned s=0;s<a.streams;s++) {
      uint64_t first=groups*s/a.streams,last=groups*(s+1)/a.streams;
      grids[s]=std::min(uint64_t(gridCap),last-first);
      if(!grids[s])continue;
      auto ap=archives.p;auto tp=lookup.p;auto rp=rawDevice.p;auto op=output.p;
      void* args[]={&ap,&stride,&tp,&rp,&op,&blocks,&a.replicas,&first,&last};
      check(hipLaunchKernel(f,dim3(grids[s]),dim3(a.threads),args,shared,streams[s]));
    }
  };
  auto synchronize=[&](){for(auto stream:streams)check(hipStreamSynchronize(stream));};
  if(!a.skip) {
    launch(true);synchronize();std::vector<uint8_t> recovered(tiled.size());
    for(unsigned r=0;r<a.replicas;r++) {
      check(hipMemcpy(recovered.data(),output.p+uint64_t(r)*tiled.size(),recovered.size(),hipMemcpyDeviceToHost));
      if(recovered!=tiled)throw std::runtime_error("actual LDS byte-exact hipANS roundtrip failed");
    }
  }
  for(unsigned i=0;i<a.warmup;i++)launch(false);synchronize();
  std::vector<double> samples;
  for(unsigned i=0;i<a.iters;i++) {
    check(hipEventRecord(begin,control));
    for(auto stream:streams)check(hipStreamWaitEvent(stream,begin,0));
    launch(false);
    for(unsigned s=0;s<a.streams;s++){check(hipEventRecord(finished[s],streams[s]));check(hipStreamWaitEvent(control,finished[s],0));}
    check(hipEventRecord(end,control));check(hipEventSynchronize(end));float ms;
    check(hipEventElapsedTime(&ms,begin,end));samples.push_back(ms);
  }
  std::sort(samples.begin(),samples.end());auto pct=[&](double p){double x=p*(samples.size()-1);size_t lo=x;return samples[lo]+(samples[std::min(lo+1,samples.size()-1)]-samples[lo])*(x-lo);};
  double median=pct(.5);if(median<=0)throw std::runtime_error("invalid latency");
  if(a.prepare) {
    std::string temporary=a.artifact+".pending";std::ofstream out(temporary,std::ios::binary);
    out.write(reinterpret_cast<char*>(archive.data()),archive.size());out.close();
    if(!out||std::rename(temporary.c_str(),a.artifact.c_str()))throw std::runtime_error("artifact save failed");
  }
  uint64_t effective=uint64_t(raw.size())*a.replicas;
  std::cout<<std::setprecision(10)<<"{\"schema\":\"hipans-lds-v1\",\"codec\":\"CURRENT hipANS ANSDecodeWarpFullBlock + decodeOneWarp, unchanged\",\"mode\":\""<<a.mode
    <<"\",\"verified_actual_LDS\":"<<(a.skip?"false":"true")<<",\"prepare\":"<<(a.prepare?"true":"false")
    <<",\"unique_weight_bytes\":"<<raw.size()<<",\"effective_decoded_bytes\":"<<effective<<",\"padded_decoded_bytes\":"<<totalBytes
    <<",\"replicas\":"<<a.replicas<<",\"artifact_bytes\":"<<archive.size()<<",\"artifact_metadata_bytes\":"<<h.getCompressedOverhead()
    <<",\"compressed_allocation_bytes\":"<<stride*a.replicas<<",\"ratio_codes\":"<<double(raw.size())/archive.size()
    <<",\"decode_ms\":{\"median\":"<<median<<",\"p10\":"<<pct(.1)<<",\"p90\":"<<pct(.9)<<"}"
    <<",\"effective_LDS_GBps\":"<<effective/median/1e6<<",\"padded_LDS_GBps\":"<<totalBytes/median/1e6
    <<",\"compressed_archive_GBps\":"<<(rawMode?0:archive.size()*a.replicas/median/1e6)
    <<",\"lookup_prepare_ms_outside_timing\":"<<tableMs<<",\"warmup\":"<<a.warmup<<",\"iters\":"<<a.iters
    <<",\"timing\":\"common start; all queues joined before stop; not sum of queue rates\""
    <<",\"replication_policy\":\"independent device addresses; same real matrix; not additional model parameters\""
    <<",\"launch\":{\"threads\":"<<a.threads<<",\"waves_per_workgroup\":"<<a.threads/64<<",\"grid_multiplier\":"<<a.gridMult<<",\"streams\":"<<a.streams<<",\"grid_per_stream\":[";
  for(unsigned i=0;i<grids.size();i++)std::cout<<(i?",":"")<<grids[i];
  std::cout<<"],\"LDS_bytes\":"<<shared<<",\"max_resident_workgroups_per_CU\":"<<activeBlocks
    <<",\"theoretical_resident_waves\":"<<uint64_t(activeBlocks)*props.multiProcessorCount*(a.threads/64)
    <<",\"tail_dummy_tiles\":"<<groups*(a.threads/64)-uint64_t(blocks)*a.replicas<<",\"VGPR\":"<<attrs.numRegs<<",\"scratch_bytes\":"<<attrs.localSizeBytes<<"}"
    <<",\"provenance\":{\"gpu_name\":\""<<props.name<<"\",\"gpu_arch\":\""<<props.gcnArchName
    <<"\",\"hipANS_commit_built\":\""<<HIPANS_COMMIT_BUILT<<"\",\"hipANS_root_built\":\""<<HIPANS_ROOT_BUILT
    <<"\",\"build_flags\":\""<<HIPANS_LDS_BUILD_FLAGS<<"\"}}\n";
  for(unsigned s=0;s<a.streams;s++){check(hipEventDestroy(finished[s]));check(hipStreamDestroy(streams[s]));}
  check(hipEventDestroy(begin));check(hipEventDestroy(end));check(hipStreamDestroy(control));
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
