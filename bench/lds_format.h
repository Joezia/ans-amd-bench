#pragma once
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <cstring>

namespace ldsans {
constexpr uint32_t Precision = 12, Lower = 1u << 23, TableSize = 1u << Precision;
struct Header {
  char magic[8];
  uint32_t version, precision, rows, cols, tile_rows, tile_cols, lanes, tiles;
  uint64_t raw_bytes, payload_bytes, reserved;
};
struct Stream { uint64_t offset; uint32_t length, state; };
static_assert(sizeof(Header) == 64 && sizeof(Stream) == 16);
struct Artifact {
  Header h{};
  std::array<uint32_t, 256> freq{};
  std::vector<Stream> streams;
  std::vector<uint8_t> payload;
};
inline std::vector<uint8_t> readBytes(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open input");
  auto size = f.tellg();
  if (size < 0) throw std::runtime_error("invalid input size");
  std::vector<uint8_t> data(static_cast<size_t>(size));
  f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!f) throw std::runtime_error("truncated input");
  return data;
}
inline std::array<uint32_t, TableSize> table(const std::array<uint32_t,256>& freq) {
  std::array<uint32_t, TableSize> lut{};
  uint32_t cdf = 0;
  for (uint32_t s=0;s<256;s++) {
    if (cdf+freq[s]>TableSize) throw std::runtime_error("invalid frequency sum");
    for (uint32_t i=0;i<freq[s];i++)
      lut[cdf+i] = s | (cdf<<8) | ((freq[s]-1)<<20);
    cdf += freq[s];
  }
  if (cdf!=TableSize) throw std::runtime_error("invalid frequency sum");
  return lut;
}
inline Artifact load(const char* path) {
  auto bytes=readBytes(path);
  if (bytes.size()<sizeof(Header)+1024) throw std::runtime_error("short artifact");
  Artifact a; std::memcpy(&a.h,bytes.data(),64);
  auto& h=a.h;
  if (std::memcmp(h.magic,"LDSANS1\0",8)||h.version!=1||h.precision!=Precision||
      h.tile_rows!=64||h.tile_cols!=128||h.lanes!=64||!h.rows||!h.cols||h.reserved)
    throw std::runtime_error("unsupported artifact header");
  uint64_t ntiles=uint64_t((h.rows+63)/64)*((uint64_t(h.cols)+127)/128);
  if (ntiles!=h.tiles || h.raw_bytes!=uint64_t(h.rows)*h.cols)
    throw std::runtime_error("invalid artifact dimensions");
  uint64_t nstreams=ntiles*h.lanes, payloadStart=64+1024+nstreams*sizeof(Stream);
  if (payloadStart>bytes.size() || h.payload_bytes!=bytes.size()-payloadStart)
    throw std::runtime_error("invalid artifact size");
  std::memcpy(a.freq.data(),bytes.data()+64,1024); table(a.freq);
  a.streams.resize(nstreams);
  std::memcpy(a.streams.data(),bytes.data()+1088,nstreams*sizeof(Stream));
  a.payload.assign(bytes.begin()+payloadStart,bytes.end());
  uint64_t end=0;
  for (auto s:a.streams) {
    if (s.offset!=end || s.length>h.payload_bytes-end || s.state<Lower)
      throw std::runtime_error("invalid stream descriptor");
    end+=s.length;
  }
  if(end!=h.payload_bytes) throw std::runtime_error("unclaimed payload bytes");
  return a;
}
inline std::vector<uint8_t> tileWeights(const std::vector<uint8_t>& raw,const Header& h) {
  if(raw.size()!=h.raw_bytes) throw std::runtime_error("raw reference size mismatch");
  std::vector<uint8_t> tiled(uint64_t(h.tiles)*8192,0);
  uint32_t columns=(h.cols+127)/128;
  for(uint32_t t=0;t<h.tiles;t++) for(uint32_t i=0;i<8192;i++) {
    uint32_t row=(t/columns)*64+i/128, col=(t%columns)*128+i%128;
    if(row<h.rows&&col<h.cols) tiled[uint64_t(t)*8192+i]=raw[uint64_t(row)*h.cols+col];
  }
  return tiled;
}
}
