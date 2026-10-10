// Offline scalar byte-renormalized rANS, 64 independent streams per GEMM tile.
#include "lds_format.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace ldsans;
int main(int argc,char**argv) { try {
  if(argc!=5) throw std::runtime_error("usage: lds_pack weights.bin rows cols output.ans");
  auto raw=readBytes(argv[1]);
  Artifact a; auto& h=a.h;
  std::memcpy(h.magic,"LDSANS1\0",8); h.version=1;h.precision=Precision;
  auto parse=[](const char* s){size_t end;auto n=std::stoull(s,&end);if(s[end]||!n||n>UINT32_MAX)throw std::runtime_error("invalid dimension");return uint32_t(n);};
  h.rows=parse(argv[2]);h.cols=parse(argv[3]);h.tile_rows=64;h.tile_cols=128;h.lanes=64;
  h.raw_bytes=uint64_t(h.rows)*h.cols;
  uint64_t nt=uint64_t((h.rows+63)/64)*((uint64_t(h.cols)+127)/128);
  if(nt>UINT32_MAX)throw std::runtime_error("too many tiles");
  h.tiles=nt;
  auto tiled=tileWeights(raw,h);
  std::array<uint64_t,256> counts{};for(auto s:tiled) counts[s]++;
  std::array<double,256> exact{};
  int sum=0;
  for(int s=0;s<256;s++) {exact[s]=double(counts[s])*TableSize/tiled.size();a.freq[s]=counts[s]?std::max(1u,uint32_t(std::floor(exact[s]))):0;sum+=a.freq[s];}
  // Same deterministic normalization rule as entropyflow-lab ref_ans.
  if(sum<int(TableSize)) {
    std::vector<int> active;for(int s=0;s<256;s++)if(counts[s])active.push_back(s);
    std::stable_sort(active.begin(),active.end(),[&](int x,int y){return exact[x]-a.freq[x]>exact[y]-a.freq[y];});
    for(int i=0;i<int(TableSize)-sum;i++)a.freq[active[i%active.size()]]++;
  } else while(sum>int(TableSize)) {
    int best=-1;for(int s=0;s<256;s++)if(a.freq[s]>1&&(best<0||a.freq[s]-exact[s]>a.freq[best]-exact[best]))best=s;
    if(best<0)throw std::runtime_error("normalization failed");
    a.freq[best]--;sum--;
  }
  std::array<uint32_t,256> cdf{};for(int s=1;s<256;s++)cdf[s]=cdf[s-1]+a.freq[s-1];
  auto lut=table(a.freq);
  for(uint32_t t=0;t<h.tiles;t++) for(uint32_t lane=0;lane<64;lane++) {
    uint32_t state=Lower;std::vector<uint8_t> emitted;
    for(int i=127;i>=0;i--) {
      auto sym=tiled[uint64_t(t)*8192+i*64+lane];auto f=a.freq[sym];
      uint32_t threshold=((Lower>>Precision)<<8)*f;
      while(state>=threshold){emitted.push_back(state&255);state>>=8;}
      state=(state/f)*TableSize+state%f+cdf[sym];
    }
    Stream stream{a.payload.size(),uint32_t(emitted.size()),state};
    std::reverse(emitted.begin(),emitted.end());
    a.payload.insert(a.payload.end(),emitted.begin(),emitted.end());a.streams.push_back(stream);
    // Independent forward decoder verifies every stream, including final state.
    uint32_t pos=0;
    for(uint32_t i=0;i<128;i++) {
      auto entry=lut[state&(TableSize-1)];auto sym=entry&255;
      if(sym!=tiled[uint64_t(t)*8192+i*64+lane])throw std::runtime_error("CPU roundtrip mismatch");
      state=((entry>>20)+1)*(state>>Precision)+(state&(TableSize-1))-((entry>>8)&4095);
      while(state<Lower){if(pos>=emitted.size())throw std::runtime_error("truncated stream");state=(state<<8)|emitted[pos++];}
    }
    if(state!=Lower||pos!=emitted.size())throw std::runtime_error("final stream state mismatch");
  }
  h.payload_bytes=a.payload.size();
  std::ofstream out(argv[4],std::ios::binary);if(!out)throw std::runtime_error("cannot write artifact");
  out.write(reinterpret_cast<char*>(&h),64);out.write(reinterpret_cast<char*>(a.freq.data()),1024);
  out.write(reinterpret_cast<char*>(a.streams.data()),a.streams.size()*sizeof(Stream));
  out.write(reinterpret_cast<char*>(a.payload.data()),a.payload.size());out.close();if(!out)throw std::runtime_error("artifact write failed");
  std::cout<<"{\"schema\":\"lds-artifact-v1\",\"cpu_roundtrip\":true,\"rows\":"<<h.rows<<",\"cols\":"<<h.cols<<",\"raw_bytes\":"<<h.raw_bytes<<",\"padded_decoded_bytes\":"<<tiled.size()<<",\"tiles\":"<<h.tiles<<",\"tile_rows\":64,\"tile_cols\":128,\"lanes\":64,\"precision\":12,\"payload_bytes\":"<<h.payload_bytes<<",\"metadata_bytes\":"<<64+1024+a.streams.size()*16<<",\"artifact_bytes\":"<<64+1024+a.streams.size()*16+h.payload_bytes<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;} }
