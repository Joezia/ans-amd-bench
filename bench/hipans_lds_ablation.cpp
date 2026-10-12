// Bench-only H3 ablations copied from the current hipANS decode path.
// Results from this executable are always INVALID as product-code performance.
#include "hipans/ans/GpuANSDecode.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef HIPANS_COMMIT_BUILT
#define HIPANS_COMMIT_BUILT "unknown"
#endif
#ifndef HIPANS_ABLATION_FLAGS
#define HIPANS_ABLATION_FLAGS "unknown"
#endif
#ifndef HIPANS_GPU_ARCH_BUILT
#define HIPANS_GPU_ARCH_BUILT "gfx942"
#endif
#ifndef HIPANS_ROCM_VERSION
#define HIPANS_ROCM_VERSION "unknown"
#endif

namespace ha = hipans;

static void check(hipError_t e) {
  if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

template <class T> struct Device {
  T* p = nullptr;
  explicit Device(size_t n) {
    check(hipMalloc(reinterpret_cast<void**>(&p), std::max(size_t(1), n) * sizeof(T)));
  }
  ~Device() { if (p) (void)hipFree(p); }
};

static std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  auto n = f.tellg();
  if (n < 0) throw std::runtime_error("invalid file length: " + path);
  std::vector<uint8_t> bytes(static_cast<size_t>(n));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!f) throw std::runtime_error("short file: " + path);
  return bytes;
}

static std::vector<uint8_t> makeTiles(const std::vector<uint8_t>& raw,
                                     uint32_t rows, uint32_t cols) {
  if (raw.size() != uint64_t(rows) * cols)
    throw std::runtime_error("reference shape mismatch");
  const uint64_t rowTiles = (uint64_t(rows) + 63) / 64;
  const uint64_t colTiles = (uint64_t(cols) + 127) / 128;
  std::vector<uint8_t> tiled(rowTiles * colTiles * ha::kDefaultBlockSize, 0);
  for (uint64_t t = 0; t < rowTiles * colTiles; ++t) {
    for (uint32_t i = 0; i < ha::kDefaultBlockSize; ++i) {
      const uint64_t row = (t / colTiles) * 64 + i / 128;
      const uint64_t col = (t % colTiles) * 128 + i % 128;
      if (row < rows && col < cols) tiled[t * ha::kDefaultBlockSize + i] = raw[row * cols + col];
    }
  }
  return tiled;
}

struct LDSWriter {
  volatile uint8_t* tile;
  bool active;
  __device__ void write(uint32_t offset, uint8_t symbol) {
    if (active) {
      uint32_t address = uint32_t(reinterpret_cast<uintptr_t>(tile + offset));
      uint32_t value = symbol;
      asm volatile("ds_write_b8 %0, %1" :: "v"(address), "v"(value) : "memory");
    }
  }
};

enum class Variant : int { A1 = 1, A2 = 2, A2c = 3, A3 = 4 };

template <Variant V> constexpr bool removesIterationBarrier() {
  return V == Variant::A1 || V == Variant::A3;
}
template <Variant V> constexpr bool stagesCompressedInput() {
  return V == Variant::A2 || V == Variant::A3;
}
template <Variant V> constexpr bool allocatesStagingLds() {
  return V != Variant::A1;
}

// Deliberate local copy of hipANS::decodeOneWarp. Only A1/A3 omit the
// workgroup barrier between the state update and wave ballot.
template <bool RemoveIterationBarrier>
__device__ inline void decodeOneWarpAblation(
    ha::ANSStateT& state, uint32_t compressedOffset,
    const ha::ANSEncodedT* __restrict__ input,
    const ha::TableT* __restrict__ lookup, uint32_t& wordsRead,
    ha::ANSDecodedT& symbol) {
  constexpr ha::ANSStateT mask = (ha::ANSStateT(1) << 10) - 1;
  const uint32_t slot = state & mask;
  uint32_t decoded, pdf, cdf;
  ha::unpackDecodeLookup(lookup[slot], decoded, pdf, cdf);
  symbol = decoded;
  state = pdf * (state >> 10) + ha::ANSStateT(cdf);
  const bool read = state < ha::kANSMinState;
  if constexpr (!RemoveIterationBarrier) __syncthreads();
  const auto vote = __ballot(read);
  const uint32_t prefix = __popcll(vote & ha::getLaneMaskGe());
  if (read) {
    const auto word = input[compressedOffset - prefix];
    state = (state << ha::kANSEncodedBits) + ha::ANSStateT(word);
  }
  wordsRead = __popcll(vote);
}

template <bool RemoveIterationBarrier, class Writer>
__device__ inline void decodeFullBlockAblation(
    unsigned lane, ha::ANSStateT state, uint32_t compressedWords,
    const ha::ANSEncodedT* __restrict__ input, Writer& writer,
    const ha::TableT* __restrict__ lookup) {
  for (int offset = ha::kDefaultBlockSize - ha::kWarpSize + lane;
       offset >= 0; offset -= ha::kWarpSize) {
    ha::ANSDecodedT symbol;
    uint32_t wordsRead;
    decodeOneWarpAblation<RemoveIterationBarrier>(
        state, compressedWords, input, lookup, wordsRead, symbol);
    compressedWords -= wordsRead;
    writer.write(offset, symbol);
  }
}

template <int Threads, bool Verify, Variant V>
__global__ __launch_bounds__(Threads) void hipansLdsAblation(
    const uint8_t* archives, uint64_t archiveStride, const ha::TableT* table,
    uint8_t* output, uint32_t blocks, uint64_t groupBegin, uint64_t groupEnd,
    uint32_t stageWordsStride) {
  extern __shared__ __align__(16) uint8_t storage[];
  constexpr unsigned waves = Threads / 64;
  auto lookup = reinterpret_cast<ha::TableT*>(storage);
  constexpr size_t tableBytes = 4096;
  constexpr size_t outputBytes = waves * ha::kDefaultBlockSize;
  constexpr size_t baseBytes = tableBytes + outputBytes;
  const unsigned tid = threadIdx.x;
  const unsigned warp = tid / 64;
  const unsigned lane = ha::getLaneId();

  auto lookup4 = reinterpret_cast<uint4*>(lookup);
  auto table4 = reinterpret_cast<const uint4*>(table);
  for (unsigned j = tid; j < 256; j += Threads) lookup4[j] = table4[j];
  __syncthreads();

  const uint64_t total = blocks;
  for (uint64_t group = groupBegin + blockIdx.x; group < groupEnd;
       group += gridDim.x) {
    const uint64_t task = group * waves + warp;
    const bool active = task < total;
    const uint64_t safeTask = active ? task : 0;
    const uint32_t block = safeTask;
    auto header = reinterpret_cast<const ha::ANSCoalescedHeader*>(archives);
    const auto states = header->getWarpStates();
    const auto blockWords = header->getBlockWords(blocks)[block];
    auto input = header->getBlockDataStart(blocks) + blockWords.y;
    const uint32_t compressedWords = blockWords.x & 0xffffU;

    if constexpr (stagesCompressedInput<V>()) {
      auto staging = reinterpret_cast<ha::ANSEncodedT*>(
          storage + baseBytes + size_t(warp) * stageWordsStride * sizeof(ha::ANSEncodedT));
      for (unsigned i = lane; i < compressedWords; i += 64) staging[i] = input[i];
      // Once per tile: publish this wave's compressed input before decoding.
      __syncthreads();
      input = staging;
    }

    LDSWriter writer{reinterpret_cast<volatile uint8_t*>(storage + tableBytes) +
                         warp * ha::kDefaultBlockSize,
                     active};
    decodeFullBlockAblation<removesIterationBarrier<V>()>(
        lane, states[block].warpState[lane], compressedWords, input,
        writer, lookup);
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    __syncthreads();
    if constexpr (Verify) {
      for (unsigned i = tid; i < waves * ha::kDefaultBlockSize; i += Threads) {
        const uint64_t owner = group * waves + i / ha::kDefaultBlockSize;
        if (owner < total)
          output[owner * ha::kDefaultBlockSize + i % ha::kDefaultBlockSize] =
              storage[tableBytes + i];
      }
      __syncthreads();
    }
  }
  (void)archiveStride;
}

template <int Threads, Variant V> static const void* kernel(bool verify) {
  if (verify)
    return reinterpret_cast<const void*>(hipansLdsAblation<Threads, true, V>);
  return reinterpret_cast<const void*>(hipansLdsAblation<Threads, false, V>);
}

static const void* selectKernel(Variant variant, bool verify) {
  switch (variant) {
    case Variant::A1: return kernel<128, Variant::A1>(verify);
    case Variant::A2: return kernel<128, Variant::A2>(verify);
    case Variant::A2c: return kernel<128, Variant::A2c>(verify);
    case Variant::A3: return kernel<128, Variant::A3>(verify);
  }
  throw std::runtime_error("unknown ablation variant");
}

static ha::ANSCoalescedHeader validateArchive(const std::vector<uint8_t>& bytes,
                                               size_t expected) {
  if (bytes.size() < 32) throw std::runtime_error("short native archive");
  ha::ANSCoalescedHeader header;
  std::memcpy(&header, bytes.data(), sizeof(header));
  uint32_t magic;
  std::memcpy(&magic, bytes.data(), sizeof(magic));
  if (magic != ((ha::kANSMagic << 16) | ha::kANSVersion) ||
      header.getProbBits() != 10 || header.getUseChecksum() ||
      header.getTotalUncompressedWords() != expected ||
      header.getNumBlocks() != expected / ha::kDefaultBlockSize ||
      header.getTotalCompressedSize() != bytes.size())
    throw std::runtime_error("artifact is not the expected native p10 hipANS archive");
  return header;
}

static uint32_t findMaxCompressedWords(const std::vector<uint8_t>& archive,
                                       uint32_t blocks) {
  const size_t offset = sizeof(ha::ANSCoalescedHeader) +
      ha::kNumSymbols * sizeof(uint16_t) + blocks * sizeof(ha::ANSWarpState);
  uint32_t maxWords = 0;
  for (uint32_t b = 0; b < blocks; ++b) {
    uint2 descriptor;
    std::memcpy(&descriptor, archive.data() + offset + b * sizeof(uint2), sizeof(descriptor));
    maxWords = std::max(maxWords, descriptor.x & 0xffffU);
  }
  return maxWords;
}

struct Options {
  std::string artifact, reference;
  uint32_t rows = 0, cols = 0, warmup = 10, iters = 50, gridMult = 4;
};

static Options parseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--artifact") o.artifact = value;
    else if (key == "--reference") o.reference = value;
    else if (key == "--rows") o.rows = std::stoul(value);
    else if (key == "--cols") o.cols = std::stoul(value);
    else if (key == "--warmup") o.warmup = std::stoul(value);
    else if (key == "--iters") o.iters = std::stoul(value);
    else if (key == "--grid-mult") o.gridMult = std::stoul(value);
    else throw std::runtime_error("unknown option " + key);
  }
  if (o.artifact.empty() || o.reference.empty() || !o.rows || !o.cols ||
      !o.iters || !o.gridMult)
    throw std::runtime_error("usage: --artifact FILE --reference FILE --rows N --cols N [--warmup N --iters N --grid-mult N]");
  return o;
}

static const char* variantName(Variant v) {
  switch (v) {
    case Variant::A1: return "A1";
    case Variant::A2: return "A2";
    case Variant::A2c: return "A2c";
    case Variant::A3: return "A3";
  }
  return "unknown";
}

struct TimedResult {
  Variant variant;
  bool verified;
  double median, p10, p90;
  double gbps;
  size_t ldsBytes;
  int residentWgs;
  uint32_t grid;
};

int main(int argc, char** argv) {
  try {
    const Options o = parseOptions(argc, argv);
    const auto raw = readFile(o.reference);
    const auto tiled = makeTiles(raw, o.rows, o.cols);
    const auto archive = readFile(o.artifact);
    const auto header = validateArchive(archive, tiled.size());
    const uint32_t blocks = header.getNumBlocks();
    const uint32_t maxCompressedWords = findMaxCompressedWords(archive, blocks);
    const uint32_t stageWordsStride = (maxCompressedWords + 7U) & ~7U;

    int device;
    check(hipGetDevice(&device));
    hipDeviceProp_t props;
    check(hipGetDeviceProperties(&props, device));
    if (props.warpSize != 64 || std::string(props.gcnArchName).rfind(HIPANS_GPU_ARCH_BUILT, 0) != 0)
      throw std::runtime_error("GPU does not match compiled wave64 target");

    hipStream_t control;
    check(hipStreamCreateWithFlags(&control, hipStreamNonBlocking));
    const uint64_t archiveStride = (archive.size() + 31) & ~uint64_t(31);
    Device<uint8_t> archives(archiveStride);
    Device<uint8_t> output(tiled.size());
    Device<ha::TableT> lookup(1024);
    check(hipMemcpyAsync(archives.p, archive.data(), archive.size(), hipMemcpyHostToDevice, control));
    ha::BatchProviderStride provider(archives.p, archiveStride);
    hipEvent_t tableBegin, tableEnd;
    check(hipEventCreate(&tableBegin));
    check(hipEventCreate(&tableEnd));
    check(hipEventRecord(tableBegin, control));
    hipLaunchKernelGGL((ha::ansDecodeTable<ha::BatchProviderStride, 512>),
                       dim3(1), dim3(512), 0, control, provider, 10, lookup.p);
    check(hipGetLastError());
    check(hipEventRecord(tableEnd, control));
    check(hipEventSynchronize(tableEnd));
    float tableMs = 0;
    check(hipEventElapsedTime(&tableMs, tableBegin, tableEnd));

    constexpr int threads = 128;
    constexpr int waves = threads / 64;
    const size_t baseLdsBytes = 4096 + waves * ha::kDefaultBlockSize;
    const size_t stageLdsBytes = size_t(waves) * stageWordsStride * sizeof(ha::ANSEncodedT);
    const std::vector<Variant> variants{Variant::A1, Variant::A2, Variant::A2c, Variant::A3};
    std::vector<TimedResult> results;
    for (Variant variant : variants) {
      const size_t extra = variant == Variant::A1 ? 0 : stageLdsBytes;
      const size_t shared = baseLdsBytes + extra;
      if (shared > props.sharedMemPerBlock)
        throw std::runtime_error(std::string(variantName(variant)) + " LDS exceeds per-workgroup limit");
      const void* timedKernel = selectKernel(variant, false);
      int activeBlocks = 0;
      check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &activeBlocks, timedKernel, threads, shared));
      if (activeBlocks <= 0) throw std::runtime_error("no resident workgroup for ablation");
      const uint64_t groups = (uint64_t(blocks) + waves - 1) / waves;
      const uint32_t gridCap = activeBlocks * props.multiProcessorCount * o.gridMult;
      const uint32_t grid = std::min<uint64_t>(gridCap, groups);
      const uint64_t first = 0, last = groups;

      auto launch = [&](bool verify) {
        const void* f = selectKernel(variant, verify);
        auto ap = archives.p;
        auto tp = lookup.p;
        auto op = output.p;
        void* args[] = {&ap, const_cast<uint64_t*>(&archiveStride), &tp,
                        &op, const_cast<uint32_t*>(&blocks),
                        const_cast<uint64_t*>(&first), const_cast<uint64_t*>(&last),
                        const_cast<uint32_t*>(&stageWordsStride)};
        check(hipLaunchKernel(f, dim3(grid), dim3(threads), args, shared, control));
        check(hipGetLastError());
      };

      launch(true);
      check(hipStreamSynchronize(control));
      std::vector<uint8_t> recovered(tiled.size());
      check(hipMemcpy(recovered.data(), output.p, recovered.size(), hipMemcpyDeviceToHost));
      const bool verified = recovered == tiled;
      if ((variant == Variant::A2 || variant == Variant::A2c) && !verified)
        throw std::runtime_error(std::string(variantName(variant)) + " required correctness verification failed");

      for (unsigned i = 0; i < o.warmup; ++i) launch(false);
      check(hipStreamSynchronize(control));
      hipEvent_t begin, end;
      check(hipEventCreate(&begin));
      check(hipEventCreate(&end));
      std::vector<double> samples;
      for (unsigned i = 0; i < o.iters; ++i) {
        check(hipEventRecord(begin, control));
        launch(false);
        check(hipEventRecord(end, control));
        check(hipEventSynchronize(end));
        float ms;
        check(hipEventElapsedTime(&ms, begin, end));
        samples.push_back(ms);
      }
      std::sort(samples.begin(), samples.end());
      auto percentile = [&](double p) {
        const double x = p * (samples.size() - 1);
        const size_t lo = static_cast<size_t>(x);
        return samples[lo] + (samples[std::min(lo + 1, samples.size() - 1)] - samples[lo]) * (x - lo);
      };
      const double median = percentile(0.5);
      results.push_back({variant, verified, median, percentile(0.1), percentile(0.9),
                         tiled.size() / median / 1.0e6, shared, activeBlocks, grid});
      check(hipEventDestroy(begin));
      check(hipEventDestroy(end));
    }

    std::cout << std::setprecision(10)
              << "{\"schema\":\"h3-ablation-t2-v1\","
              << "\"result_validity\":\"INVALID: bench-only ablation; not product codec performance\","
              << "\"verification_scope\":\"byte-exact LDS output checked outside timed region; A2/A2c required to pass\","
              << "\"warmup\":" << o.warmup << ",\"iters\":" << o.iters
              << ",\"workload\":{\"raw_bytes\":" << raw.size()
              << ",\"artifact_bytes\":" << archive.size()
              << ",\"artifact_metadata_bytes\":" << header.getCompressedOverhead()
              << ",\"blocks\":" << blocks << ",\"max_compressed_words_per_block\":" << maxCompressedWords
              << ",\"stage_words_stride_per_wave\":" << stageWordsStride
              << ",\"replicas\":1,\"shape\":[" << o.rows << "," << o.cols << "]},"
              << "\"launch_common\":{\"threads\":128,\"waves_per_workgroup\":2,\"grid_multiplier\":"
              << o.gridMult << ",\"streams\":1,\"table_prepare_ms_outside_timing\":" << tableMs << "},"
              << "\"provenance\":{\"gpu_name\":\"" << props.name
              << "\",\"gpu_arch\":\"" << props.gcnArchName
              << "\",\"ROCm_version\":\"" << HIPANS_ROCM_VERSION
              << "\",\"hipANS_commit\":\"" << HIPANS_COMMIT_BUILT
              << "\",\"build_flags\":\"" << HIPANS_ABLATION_FLAGS << "\"},\"variants\":[";
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      if (i) std::cout << ',';
      std::cout << "{\"id\":\"" << variantName(r.variant)
                << "\",\"result_validity\":\"INVALID: bench-only ablation\","
                << "\"verified_output\":" << (r.verified ? "true" : "false")
                << ",\"decode_ms\":{\"median\":" << r.median << ",\"p10\":" << r.p10
                << ",\"p90\":" << r.p90 << "},\"effective_LDS_GBps\":" << r.gbps
                << ",\"dynamic_LDS_bytes\":" << r.ldsBytes
                << ",\"resident_workgroups_per_CU\":" << r.residentWgs
                << ",\"grid\":" << r.grid << '}';
    }
    std::cout << "]}\n";
    check(hipEventDestroy(tableBegin));
    check(hipEventDestroy(tableEnd));
    check(hipStreamDestroy(control));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
