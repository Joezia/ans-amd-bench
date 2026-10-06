#include <glog/logging.h>

#include "hipans/ans/GpuANSCodec.h"
#include "hipans/ans/GpuANSUtils.h"
#include "hipans/utils/StackDeviceMemory.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <utility>
#include <vector>

#ifndef HIPANS_SOURCE_DIR
#define HIPANS_SOURCE_DIR "unknown"
#endif
#ifndef HIPANS_GPU_ARCH
#define HIPANS_GPU_ARCH "unknown"
#endif
#ifndef ANS_BENCH_BUILD_FLAGS
#define ANS_BENCH_BUILD_FLAGS "unknown"
#endif

namespace {

constexpr uint64_t kDefaultStackBytes = 4ULL * 1024 * 1024 * 1024;
#ifndef HIPANS_DECODE_THREADS
#define HIPANS_DECODE_THREADS 128
#endif
#ifndef HIPANS_GRID_OVERSUBSCRIPTION
#define HIPANS_GRID_OVERSUBSCRIPTION 4
#endif
constexpr uint32_t kDecodeBlockThreads = HIPANS_DECODE_THREADS;
constexpr uint32_t kDecodeTableBlockThreads = 512;
constexpr uint32_t kGridOversubscription = HIPANS_GRID_OVERSUBSCRIPTION;
constexpr uint32_t kAnsStreamsPerWorkgroup = HIPANS_DECODE_THREADS;

struct Options {
  std::string input;
  std::string meta;
  int warmup = 10;
  int iters = 50;
  int prob_bits = 10;
  double peak_gbps = 0.0;
};

struct Metadata {
  uint64_t num_elems = 0;
  uint64_t bits_per_symbol = 0;
  uint64_t tile_size = 0;
};

struct Stats {
  double median = 0.0;
  double p10 = 0.0;
  double p90 = 0.0;
};

struct CommandResult {
  int exit_code = -1;
  std::string output;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      const hipError_t status = hipFree(ptr_);
      if (status != hipSuccess) {
        std::cerr << "WARNING: hipFree failed: " << hipGetErrorString(status)
                  << "\n";
      }
    }
  }

  bool allocate(size_t count, const char* label) {
    count_ = count;
    if (count == 0) {
      std::cerr << "ERROR: zero-size allocation for " << label << "\n";
      return false;
    }
    const hipError_t status =
        hipMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T));
    if (status != hipSuccess) {
      std::cerr << "ERROR: hipMalloc(" << label << ", "
                << count * sizeof(T) << " bytes): "
                << hipGetErrorString(status) << "\n";
      ptr_ = nullptr;
      count_ = 0;
      return false;
    }
    return true;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t count() const { return count_; }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

bool checkHip(hipError_t status, const char* operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << "ERROR: " << operation << ": " << hipGetErrorString(status)
            << "\n";
  return false;
}

bool parseInteger(const std::string& text, uint64_t* result) {
  try {
    size_t consumed = 0;
    *result = std::stoull(text, &consumed, 0);
    return consumed == text.size();
  } catch (...) {
    return false;
  }
}

bool parseSigned(const std::string& text, int* result) {
  try {
    size_t consumed = 0;
    *result = std::stoi(text, &consumed, 0);
    return consumed == text.size();
  } catch (...) {
    return false;
  }
}

bool parseDouble(const std::string& text, double* result) {
  try {
    size_t consumed = 0;
    *result = std::stod(text, &consumed);
    return consumed == text.size() && std::isfinite(*result);
  } catch (...) {
    return false;
  }
}

void usage(const char* program) {
  std::cerr
      << "Usage: " << program
      << " --input symbols.bin --meta meta.json [--warmup N --iters N "
         "--peak-GBps X]\n"
      << "Compatibility: --symbols aliases --input; --json is accepted and "
         "is always enabled.\n"
      << "Advanced: --prob-bits 9|10|11 (default 10).\n";
}

bool parseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (arg == "--json") {
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "ERROR: missing value after " << arg << "\n";
      return false;
    }
    const std::string value = argv[++i];
    if (arg == "--input" || arg == "--symbols") {
      options->input = value;
    } else if (arg == "--meta") {
      options->meta = value;
    } else if (arg == "--warmup") {
      if (!parseSigned(value, &options->warmup)) return false;
    } else if (arg == "--iters") {
      if (!parseSigned(value, &options->iters)) return false;
    } else if (arg == "--peak-GBps") {
      if (!parseDouble(value, &options->peak_gbps)) return false;
    } else if (arg == "--prob-bits") {
      if (!parseSigned(value, &options->prob_bits)) return false;
    } else {
      std::cerr << "ERROR: unknown option " << arg << "\n";
      return false;
    }
  }
  if (options->input.empty() || options->meta.empty()) {
    std::cerr << "ERROR: --input and --meta are required\n";
    return false;
  }
  if (options->warmup < 0 || options->iters <= 0) {
    std::cerr << "ERROR: --warmup must be nonnegative and --iters positive\n";
    return false;
  }
  if (options->prob_bits < 9 || options->prob_bits > 11) {
    std::cerr << "ERROR: --prob-bits must be 9, 10, or 11\n";
    return false;
  }
  if (options->peak_gbps < 0.0) {
    std::cerr << "ERROR: --peak-GBps must be nonnegative\n";
    return false;
  }
  return true;
}

bool readFile(const std::string& path, std::vector<uint8_t>* data) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    std::cerr << "ERROR: cannot open " << path << "\n";
    return false;
  }
  const std::streamoff size = input.tellg();
  if (size <= 0 || static_cast<uint64_t>(size) >
                       static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    std::cerr << "ERROR: input must be nonempty and fit in host memory\n";
    return false;
  }
  input.seekg(0, std::ios::beg);
  data->resize(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(data->data()), size);
  if (!input) {
    std::cerr << "ERROR: failed while reading " << path << "\n";
    return false;
  }
  return true;
}

bool readText(const std::string& path, std::string* text) {
  std::ifstream input(path);
  if (!input) {
    std::cerr << "ERROR: cannot open " << path << "\n";
    return false;
  }
  std::ostringstream out;
  out << input.rdbuf();
  *text = out.str();
  return input.good() || input.eof();
}

bool jsonUnsigned(const std::string& json, const std::string& key,
                  uint64_t* value) {
  const std::regex expression("\\\"" + key +
                              "\\\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (!std::regex_search(json, match, expression)) {
    std::cerr << "ERROR: meta.json requires integer field " << key << "\n";
    return false;
  }
  return parseInteger(match[1].str(), value);
}

bool parseMetadata(const std::string& json, Metadata* meta) {
  if (!jsonUnsigned(json, "num_elems", &meta->num_elems) ||
      !jsonUnsigned(json, "bits_per_symbol", &meta->bits_per_symbol) ||
      !jsonUnsigned(json, "tile_size", &meta->tile_size)) {
    return false;
  }
  if (meta->num_elems == 0 || meta->bits_per_symbol == 0 ||
      meta->tile_size == 0) {
    std::cerr << "ERROR: metadata values must be positive\n";
    return false;
  }
  return true;
}

uint64_t stackBytes() {
  const char* value = std::getenv("HIPANS_STACK_BYTES");
  if (value == nullptr || *value == '\0') return kDefaultStackBytes;
  uint64_t parsed = 0;
  if (!parseInteger(value, &parsed) || parsed < 64ULL * 1024 * 1024) {
    std::cerr << "ERROR: HIPANS_STACK_BYTES must be an integer >= 64 MiB\n";
    return 0;
  }
  return parsed;
}

uint64_t ceilDiv(uint64_t value, uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

bool makeTileSizes(const Metadata& meta, size_t file_bytes,
                   std::vector<uint32_t>* sizes, std::string* layout) {
  if (meta.num_elems > std::numeric_limits<uint64_t>::max() /
                           meta.bits_per_symbol) {
    std::cerr << "ERROR: num_elems * bits_per_symbol overflows\n";
    return false;
  }
  const uint64_t packed_bytes = ceilDiv(meta.num_elems * meta.bits_per_symbol, 8);
  const bool expanded_u8 = file_bytes == meta.num_elems &&
                           meta.bits_per_symbol <= 8 &&
                           file_bytes != packed_bytes;
  if (file_bytes != packed_bytes && !expanded_u8) {
    std::cerr << "ERROR: symbols.bin has " << file_bytes
              << " bytes; expected packed size " << packed_bytes
              << " from metadata, or one uint8 per symbol (" << meta.num_elems
              << " bytes)\n";
    return false;
  }
  *layout = expanded_u8 ? "expanded_uint8_symbols" : "packed_bits";

  const uint64_t tiles = ceilDiv(meta.num_elems, meta.tile_size);
  if (tiles > std::numeric_limits<uint32_t>::max()) {
    std::cerr << "ERROR: tile count exceeds hipANS batch limit\n";
    return false;
  }
  if (!expanded_u8 && (meta.tile_size * meta.bits_per_symbol) % 8 != 0 &&
      tiles > 1) {
    std::cerr << "ERROR: packed tile boundaries must be byte aligned: "
              << "tile_size * bits_per_symbol must be divisible by 8\n";
    return false;
  }
  sizes->reserve(static_cast<size_t>(tiles));
  uint64_t symbols_left = meta.num_elems;
  uint64_t bytes_seen = 0;
  while (symbols_left != 0) {
    const uint64_t symbols = std::min(symbols_left, meta.tile_size);
    const uint64_t bytes = expanded_u8 ? symbols
                                       : ceilDiv(symbols * meta.bits_per_symbol, 8);
    if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
      std::cerr << "ERROR: each tile must contain 1..UINT32_MAX bytes\n";
      return false;
    }
    sizes->push_back(static_cast<uint32_t>(bytes));
    bytes_seen += bytes;
    symbols_left -= symbols;
  }
  if (bytes_seen != file_bytes) {
    std::cerr << "ERROR: tiled byte count " << bytes_seen
              << " differs from file size " << file_bytes << "\n";
    return false;
  }
  return true;
}

double percentile(const std::vector<double>& sorted, double fraction) {
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Stats summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return Stats{percentile(values, 0.5), percentile(values, 0.1),
               percentile(values, 0.9)};
}

CommandResult command(const std::string& value) {
  CommandResult result;
  FILE* pipe = popen((value + " 2>/dev/null").c_str(), "r");
  if (pipe == nullptr) return result;
  char buffer[1024];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) result.output += buffer;
  const int status = pclose(pipe);
  if (status != -1 && WIFEXITED(status)) result.exit_code = WEXITSTATUS(status);
  while (!result.output.empty() &&
         (result.output.back() == '\n' || result.output.back() == '\r')) {
    result.output.pop_back();
  }
  return result;
}

std::string jsonEscape(const std::string& value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned int>(c) << std::dec;
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

std::string rocmVersion() {
  for (const char* path : {"/opt/rocm/.info/version", "/opt/rocm/.info/version-dev"}) {
    std::ifstream input(path);
    std::string version;
    if (input && std::getline(input, version) && !version.empty()) return version;
  }
  int runtime = 0;
  if (hipRuntimeGetVersion(&runtime) == hipSuccess) return std::to_string(runtime);
  return "unknown";
}

std::pair<std::string, bool> gitProvenance() {
  const std::string root = HIPANS_SOURCE_DIR;
  const CommandResult commit = command("git -C \"" + root + "\" rev-parse HEAD");
  const CommandResult dirty = command("git -C \"" + root + "\" status --porcelain");
  return {commit.exit_code == 0 ? commit.output : "unknown",
          dirty.exit_code != 0 || !dirty.output.empty()};
}

uint64_t metadataBytes(const std::vector<uint32_t>& tile_sizes) {
  uint64_t total = 0;
  for (const uint32_t bytes : tile_sizes) {
    const uint32_t blocks = static_cast<uint32_t>(ceilDiv(bytes, hipans::kDefaultBlockSize));
    total += hipans::ANSCoalescedHeader::getCompressedOverhead(blocks);
  }
  return total;
}

bool verify(const std::vector<uint8_t>& expected, uint8_t* output_device,
            hipStream_t stream) {
  std::vector<uint8_t> actual(expected.size());
  if (!checkHip(hipMemcpyAsync(actual.data(), output_device, actual.size(),
                               hipMemcpyDeviceToHost, stream),
                "roundtrip D2H") ||
      !checkHip(hipStreamSynchronize(stream), "roundtrip synchronize")) {
    return false;
  }
  if (actual != expected) {
    size_t offset = 0;
    while (offset < actual.size() && actual[offset] == expected[offset]) ++offset;
    std::cerr << "ERROR: byte-exact verification failed at offset " << offset
              << "\n";
    return false;
  }
  return true;
}

int run(int argc, char** argv) {
  Options options;
  if (!parseOptions(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }

  std::vector<uint8_t> input;
  std::string metadata_text;
  Metadata metadata;
  if (!readFile(options.input, &input) ||
      !readText(options.meta, &metadata_text) ||
      !parseMetadata(metadata_text, &metadata)) {
    return 3;
  }

  std::vector<uint32_t> tile_sizes;
  std::string input_layout;
  if (!makeTileSizes(metadata, input.size(), &tile_sizes, &input_layout)) return 3;
  const uint32_t num_tiles = static_cast<uint32_t>(tile_sizes.size());
  const uint32_t max_tile_bytes =
      *std::max_element(tile_sizes.begin(), tile_sizes.end());
  const uint32_t compressed_stride = hipans::getMaxCompressedSize(max_tile_bytes);
  if (static_cast<uint64_t>(compressed_stride) * num_tiles >
      std::numeric_limits<size_t>::max()) {
    std::cerr << "ERROR: compressed allocation size overflows size_t\n";
    return 4;
  }

  int device = 0;
  hipDeviceProp_t properties{};
  if (!checkHip(hipGetDevice(&device), "hipGetDevice") ||
      !checkHip(hipGetDeviceProperties(&properties, device),
                "hipGetDeviceProperties")) {
    return 4;
  }
  const std::string runtime_arch = properties.gcnArchName;
  if (runtime_arch.rfind(HIPANS_GPU_ARCH, 0) != 0) {
    std::cerr << "ERROR: binary target " << HIPANS_GPU_ARCH
              << " does not match runtime architecture " << runtime_arch << "\n";
    return 4;
  }

  DeviceBuffer<uint8_t> input_device;
  DeviceBuffer<uint8_t> compressed_device;
  DeviceBuffer<uint8_t> output_device;
  DeviceBuffer<uint32_t> compressed_sizes_device;
  DeviceBuffer<uint8_t> success_device;
  DeviceBuffer<uint32_t> output_sizes_device;
  if (!input_device.allocate(input.size(), "input") ||
      !compressed_device.allocate(
          static_cast<size_t>(compressed_stride) * num_tiles, "compressed") ||
      !output_device.allocate(input.size(), "output") ||
      !compressed_sizes_device.allocate(num_tiles, "compressed sizes") ||
      !success_device.allocate(num_tiles, "decode status") ||
      !output_sizes_device.allocate(num_tiles, "decode sizes")) {
    return 4;
  }

  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "create stream") ||
      !checkHip(hipEventCreate(&start), "create start event") ||
      !checkHip(hipEventCreate(&stop), "create stop event")) {
    return 4;
  }

  int result = 0;
  double encode_ms = 0.0;
  std::vector<double> samples;
  uint64_t compressed_bytes = 0;
  const hipans::ANSCodecConfig config(options.prob_bits, false);
  const uint64_t stack_bytes = stackBytes();
  if (stack_bytes == 0) return 4;
  {
    auto stack = hipans::makeStackMemory(stack_bytes);
    if (!checkHip(hipMemcpyAsync(input_device.data(), input.data(), input.size(),
                                 hipMemcpyHostToDevice, stream), "input H2D") ||
        !checkHip(hipStreamSynchronize(stream), "input H2D synchronize")) {
      result = 5;
    }

    std::vector<const void*> input_ptrs(num_tiles);
    std::vector<const void*> compressed_input_ptrs(num_tiles);
    std::vector<void*> compressed_output_ptrs(num_tiles);
    std::vector<void*> output_ptrs(num_tiles);
    size_t offset = 0;
    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
      input_ptrs[tile] = input_device.data() + offset;
      output_ptrs[tile] = output_device.data() + offset;
      compressed_input_ptrs[tile] =
          compressed_device.data() + static_cast<size_t>(tile) * compressed_stride;
      compressed_output_ptrs[tile] =
          compressed_device.data() + static_cast<size_t>(tile) * compressed_stride;
      offset += tile_sizes[tile];
    }

    if (result == 0 &&
        (!checkHip(hipEventRecord(start, stream), "record encode start"))) {
      result = 5;
    }
    if (result == 0) {
      hipans::ansEncodeBatchPointer(
          stack, config, num_tiles, input_ptrs.data(), tile_sizes.data(), nullptr,
          compressed_output_ptrs.data(), compressed_sizes_device.data(), stream);
      if (!checkHip(hipEventRecord(stop, stream), "record encode stop") ||
          !checkHip(hipEventSynchronize(stop), "wait encode stop")) {
        result = 5;
      }
    }
    float encode_elapsed = 0.0f;
    if (result == 0 &&
        !checkHip(hipEventElapsedTime(&encode_elapsed, start, stop),
                  "encode elapsed time")) {
      result = 5;
    }
    encode_ms = encode_elapsed;

    std::vector<uint32_t> compressed_sizes(num_tiles);
    if (result == 0 &&
        !checkHip(hipMemcpy(compressed_sizes.data(), compressed_sizes_device.data(),
                            compressed_sizes.size() * sizeof(uint32_t),
                            hipMemcpyDeviceToHost), "compressed sizes D2H")) {
      result = 5;
    }
    for (uint32_t tile = 0; result == 0 && tile < num_tiles; ++tile) {
      if (compressed_sizes[tile] == 0 ||
          compressed_sizes[tile] > compressed_stride) {
        std::cerr << "ERROR: invalid compressed size for tile " << tile << "\n";
        result = 5;
      }
      compressed_bytes += compressed_sizes[tile];
    }

    auto decode_once = [&]() -> bool {
      const hipans::ANSDecodeStatus status = hipans::ansDecodeBatchPointer(
          stack, config, num_tiles, compressed_input_ptrs.data(),
          output_ptrs.data(), tile_sizes.data(), success_device.data(),
          output_sizes_device.data(), stream);
      return status.error == hipans::ANSDecodeError::None;
    };

    if (result == 0) {
      if (!decode_once() ||
          !checkHip(hipStreamSynchronize(stream), "correctness decode")) {
        std::cerr << "ERROR: correctness decode failed\n";
        result = 6;
      }
    }
    if (result == 0 && !verify(input, output_device.data(), stream)) result = 6;

    for (int i = 0; result == 0 && i < options.warmup; ++i) {
      if (!decode_once() ||
          !checkHip(hipStreamSynchronize(stream), "warmup decode")) {
        std::cerr << "ERROR: warmup decode failed at iteration " << i << "\n";
        result = 7;
      }
    }

    samples.reserve(options.iters);
    for (int i = 0; result == 0 && i < options.iters; ++i) {
      if (!checkHip(hipEventRecord(start, stream), "record decode start") ||
          !decode_once() ||
          !checkHip(hipEventRecord(stop, stream), "record decode stop") ||
          !checkHip(hipEventSynchronize(stop), "wait decode stop")) {
        std::cerr << "ERROR: measured decode failed at iteration " << i << "\n";
        result = 7;
        break;
      }
      float elapsed = 0.0f;
      if (!checkHip(hipEventElapsedTime(&elapsed, start, stop),
                    "decode elapsed time")) {
        result = 7;
        break;
      }
      samples.push_back(elapsed);
    }
    if (result == 0 && !verify(input, output_device.data(), stream)) result = 7;
    std::cerr << "stack_max_memory_usage_bytes=" << stack.getMaxMemoryUsage()
              << "\n";
  }

  if (!checkHip(hipEventDestroy(start), "destroy start event") && result == 0)
    result = 8;
  if (!checkHip(hipEventDestroy(stop), "destroy stop event") && result == 0)
    result = 8;
  if (!checkHip(hipStreamDestroy(stream), "destroy stream") && result == 0)
    result = 8;
  if (result != 0) return result;

  const Stats stats = summarize(samples);
  const double seconds = stats.median / 1000.0;
  const double compressed_gbps = compressed_bytes / seconds / 1.0e9;
  const double decoded_gbps = input.size() / seconds / 1.0e9;
  const double utilization = options.peak_gbps > 0.0
                                 ? decoded_gbps / options.peak_gbps
                                 : 0.0;
  const uint64_t table_metadata_bytes = metadataBytes(tile_sizes);
  const double ratio = static_cast<double>(input.size()) /
                       static_cast<double>(compressed_bytes);
  const auto provenance = gitProvenance();

  const uint64_t max_blocks_per_tile = ceilDiv(max_tile_bytes, hipans::kDefaultBlockSize);
  std::cout << std::setprecision(10)
            << "{"
            << "\"raw_bytes\":" << input.size() << ','
            << "\"compressed_bytes\":" << compressed_bytes << ','
            << "\"table_metadata_bytes\":" << table_metadata_bytes << ','
            << "\"ratio\":" << ratio << ','
            << "\"encode_ms\":" << encode_ms << ','
            << "\"decode_ms\":{\"median\":" << stats.median
            << ",\"p10\":" << stats.p10 << ",\"p90\":" << stats.p90
            << "},"
            << "\"decode_GBps_over_compressed_read\":" << compressed_gbps
            << ','
            << "\"decode_GBps_over_decoded_written\":" << decoded_gbps << ','
            << "\"utilization_vs_peak\":";
  if (options.peak_gbps > 0.0) std::cout << utilization;
  else std::cout << "null";
  std::cout << ",\"provenance\":{"
            << "\"hipANS_git_commit\":\"" << jsonEscape(provenance.first)
            << "\",\"hipANS_dirty\":" << (provenance.second ? "true" : "false")
            << ",\"build_flags\":\"" << jsonEscape(ANS_BENCH_BUILD_FLAGS)
            << "\",\"ROCm_version\":\"" << jsonEscape(rocmVersion())
            << "\",\"GPU_name\":\"" << jsonEscape(properties.name)
            << "\",\"GPU_arch\":\"" << jsonEscape(runtime_arch) << "\""
            << ",\"launch_config\":{"
            << "\"grid\":{\"x\":\"ceil(occupancy_blocks/" << num_tiles
            << ")*" << kGridOversubscription << "\",\"y\":" << num_tiles
            << "},\"block\":{\"decode\":" << kDecodeBlockThreads
            << ",\"table\":" << kDecodeTableBlockThreads << "},"
            << "\"streams_per_workgroup\":" << kAnsStreamsPerWorkgroup
            << ",\"hip_streams\":1,\"rans_block_bytes\":"
            << hipans::kDefaultBlockSize << ",\"max_rans_blocks_per_tile\":"
            << max_blocks_per_tile << "},"
            << "\"probability_bits\":" << options.prob_bits << ','
            << "\"num_tiles\":" << num_tiles << ','
            << "\"input_layout\":\"" << input_layout << "\""
            << "}}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  return run(argc, argv);
}
