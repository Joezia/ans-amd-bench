#include <glog/logging.h>

#include "hipans/ans/GpuANSCodec.h"
#include "hipans/utils/StackDeviceMemory.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr uint64_t kDefaultBlockSize = 8192;
constexpr uint64_t kStackBytes = 2ULL * 1024 * 1024 * 1024;
constexpr const char* kBuildFlags =
    "-std=c++17 -O3 -DNDEBUG --offload-arch=gfx90a -pthread";
#ifndef HIPANS_SOURCE_DIR
#define HIPANS_SOURCE_DIR "unknown"
#endif
#ifndef BENCH_SOURCE_DIR
#define BENCH_SOURCE_DIR "unknown"
#endif
constexpr const char* kRepo = HIPANS_SOURCE_DIR;
constexpr const char* kBenchDir = BENCH_SOURCE_DIR;
constexpr const char* kOverflowWarning =
    "StackDeviceMemory: attempting to allocate";

struct Options {
  std::string dataset;
  std::string file;
  std::string csv;
  std::string case_label = "unspecified";
  uint64_t size_bytes = 0;
  uint64_t seed = 20260920;
  double lambda = 100.0;
  int prob_bits = 10;
  uint32_t num_in_batch = 1;
  int warmup_iters = 10;
  int measure_iters = 50;
  int gpu_index = 0;
  bool verify_checksum_contract = false;
};

struct CommandResult {
  int exit_code = -1;
  std::string output;
};

struct GpuSample {
  std::string label;
  std::string raw;
  std::optional<double> utilization;
  std::optional<double> sclk_mhz;
  std::optional<double> mclk_mhz;
  std::optional<double> temperature_c;
  bool command_ok = false;
};

struct Stats {
  double median = 0.0;
  double p5 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

class TeeStreamBuf final : public std::streambuf {
 public:
  explicit TeeStreamBuf(std::streambuf* original) : original_(original) {}

  std::string captured() const { return capture_.str(); }

 protected:
  int overflow(int ch) override {
    if (ch == traits_type::eof()) {
      return traits_type::not_eof(ch);
    }
    const char c = static_cast<char>(ch);
    if (original_->sputc(c) == traits_type::eof()) {
      return traits_type::eof();
    }
    capture_.put(c);
    return ch;
  }

  std::streamsize xsputn(const char* data, std::streamsize count) override {
    const auto written = original_->sputn(data, count);
    capture_.write(data, written);
    return written;
  }

  int sync() override {
    capture_.flush();
    return original_->pubsync();
  }

 private:
  std::streambuf* original_;
  std::ostringstream capture_;
};

class ScopedStderrCapture final {
 public:
  ScopedStderrCapture() : old_(std::cerr.rdbuf()), tee_(old_) {
    std::cerr.rdbuf(&tee_);
  }

  ~ScopedStderrCapture() {
    std::cerr.flush();
    std::cerr.rdbuf(old_);
  }

  std::string captured() const { return tee_.captured(); }

 private:
  std::streambuf* old_;
  TeeStreamBuf tee_;
};

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      const hipError_t error = hipFree(ptr_);
      if (error != hipSuccess) {
        std::cerr << "ERROR: hipFree failed for " << name_ << ": "
                  << hipGetErrorString(error) << "\n";
      }
    }
  }

  bool allocate(size_t count, const std::string& name) {
    name_ = name;
    count_ = count;
    if (count == 0) {
      std::cerr << "ERROR: zero-sized device allocation requested for " << name
                << "\n";
      return false;
    }
    const hipError_t error = hipMalloc(reinterpret_cast<void**>(&ptr_),
                                       count * sizeof(T));
    if (error != hipSuccess) {
      std::cerr << "ERROR: hipMalloc failed for " << name << " ("
                << count * sizeof(T) << " bytes): "
                << hipGetErrorString(error) << "\n";
      ptr_ = nullptr;
      count_ = 0;
      return false;
    }
    return true;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t count() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  std::string name_;
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

bool checkHip(hipError_t error, const char* operation) {
  if (error == hipSuccess) {
    return true;
  }
  std::cerr << "ERROR: " << operation << " failed: "
            << hipGetErrorString(error) << "\n";
  return false;
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

CommandResult runCommand(const std::string& command) {
  CommandResult result;
  const std::string wrapped = command + " 2>&1";
  FILE* pipe = popen(wrapped.c_str(), "r");
  if (pipe == nullptr) {
    result.output = "popen failed";
    return result;
  }

  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result.output += buffer;
  }

  const int status = pclose(pipe);
  if (status != -1 && WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  }
  return result;
}

std::string commandValue(const std::string& command,
                         const std::string& fallback) {
  const CommandResult result = runCommand(command);
  const std::string value = trim(result.output);
  if (result.exit_code != 0 || value.empty()) {
    return fallback;
  }
  return value;
}

std::optional<double> numberFromLine(const std::string& text,
                                     const std::string& required,
                                     const std::string& unit_pattern) {
  std::istringstream input(text);
  std::string line;
  const std::regex value_regex("([0-9]+(?:\\.[0-9]+)?)\\s*" + unit_pattern,
                               std::regex::icase);
  while (std::getline(input, line)) {
    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lowered.find(required) == std::string::npos) {
      continue;
    }
    std::smatch match;
    if (std::regex_search(line, match, value_regex)) {
      return std::stod(match[1].str());
    }
    const auto colon = line.find_last_of(':');
    if (colon != std::string::npos) {
      const std::regex plain_regex("([0-9]+(?:\\.[0-9]+)?)");
      const std::string suffix = line.substr(colon + 1);
      if (std::regex_search(suffix, match, plain_regex)) {
        return std::stod(match[1].str());
      }
    }
  }
  return std::nullopt;
}

std::optional<double> utilizationFromOutput(const std::string& text) {
  std::istringstream input(text);
  std::string line;
  while (std::getline(input, line)) {
    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lowered.find("gpu use") == std::string::npos &&
        lowered.find("gpu%") == std::string::npos) {
      continue;
    }
    const std::regex value_regex("([0-9]+(?:\\.[0-9]+)?)\\s*%");
    std::smatch match;
    if (std::regex_search(line, match, value_regex)) {
      return std::stod(match[1].str());
    }
    const auto colon = line.find_last_of(':');
    if (colon != std::string::npos) {
      const std::regex plain_regex("([0-9]+(?:\\.[0-9]+)?)");
      const std::string suffix = line.substr(colon + 1);
      if (std::regex_search(suffix, match, plain_regex)) {
        return std::stod(match[1].str());
      }
    }
  }
  return std::nullopt;
}

GpuSample sampleGpu(const std::string& label, int physical_gpu) {
  const std::string command =
      "rocm-smi -d " + std::to_string(physical_gpu) +
      " --showuse --showclocks --showtemp";
  const CommandResult result = runCommand(command);

  GpuSample sample;
  sample.label = label;
  sample.raw = result.output;
  sample.command_ok = result.exit_code == 0;
  sample.utilization = utilizationFromOutput(result.output);
  sample.sclk_mhz = numberFromLine(result.output, "sclk", "mhz");
  sample.mclk_mhz = numberFromLine(result.output, "mclk", "mhz");
  sample.temperature_c = numberFromLine(result.output, "temperature", "(?:c|°c)");

  std::cout << "----- rocm-smi sample: " << label << " -----\n";
  std::cout << "$ " << command << "\n";
  std::cout << result.output;
  if (!result.output.empty() && result.output.back() != '\n') {
    std::cout << '\n';
  }
  std::cout << "parsed: util=";
  if (sample.utilization) {
    std::cout << *sample.utilization << "%";
  } else {
    std::cout << "NA";
  }
  std::cout << " sclk=";
  if (sample.sclk_mhz) {
    std::cout << *sample.sclk_mhz << "MHz";
  } else {
    std::cout << "NA";
  }
  std::cout << " mclk=";
  if (sample.mclk_mhz) {
    std::cout << *sample.mclk_mhz << "MHz";
  } else {
    std::cout << "NA";
  }
  std::cout << " temp=";
  if (sample.temperature_c) {
    std::cout << *sample.temperature_c << "C";
  } else {
    std::cout << "NA";
  }
  std::cout << " command_rc=" << result.exit_code << "\n";
  return sample;
}

std::string optionalNumber(const std::optional<double>& value,
                           int precision = 3) {
  if (!value) {
    return "NA";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << *value;
  return out.str();
}

std::string fixed(double value, int precision = 6) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string isoTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t value = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&value, &utc);
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::string hostname() {
  char value[256] = {};
  if (gethostname(value, sizeof(value) - 1) != 0) {
    return "unknown";
  }
  return value;
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (const char c : value) {
    if (c == '\"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += '\"';
  return escaped;
}

bool appendCsv(const std::string& path,
               const std::vector<std::string>& values) {
  static const std::vector<std::string> header = {
      "timestamp", "hostname", "gpu_index", "gpu_arch", "rocm_version",
      "hipans_commit", "harness_commit", "build_flags", "dataset",
      "probbits", "num_in_batch", "size_bytes", "uncompressed_bytes",
      "compressed_bytes", "compression_ratio", "correctness", "run_status",
      "warmup_iters", "measure_iters", "kernel_ms_median", "kernel_ms_p5",
      "kernel_ms_p95", "kernel_ms_min", "kernel_ms_max", "wall_ms_median",
      "throughput_GBps_uncompressed", "throughput_GBps_compressed",
      "throughput_GBps_total_traffic", "sclk_begin", "sclk_mid", "sclk_end",
      "temp_begin", "temp_end", "notes"};

  if (values.size() != header.size()) {
    std::cerr << "ERROR: internal CSV column mismatch: values=" << values.size()
              << " header=" << header.size() << "\n";
    return false;
  }

  bool needs_header = true;
  std::error_code error;
  if (std::filesystem::exists(path, error)) {
    needs_header = std::filesystem::file_size(path, error) == 0;
  }

  std::ofstream output(path, std::ios::app);
  if (!output) {
    std::cerr << "ERROR: cannot append CSV: " << path << "\n";
    return false;
  }

  const auto write_row = [&output](const std::vector<std::string>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i != 0) {
        output << ',';
      }
      output << csvEscape(row[i]);
    }
    output << '\n';
  };

  if (needs_header) {
    write_row(header);
  }
  write_row(values);
  output.flush();
  return output.good();
}

double percentile(const std::vector<double>& sorted, double fraction) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double position = fraction * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

Stats calculateStats(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  Stats stats;
  stats.median = percentile(values, 0.50);
  stats.p5 = percentile(values, 0.05);
  stats.p95 = percentile(values, 0.95);
  stats.min = values.front();
  stats.max = values.back();
  return stats;
}

bool parseUnsigned(const std::string& text, uint64_t* value) {
  try {
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(text, &consumed, 0);
    if (consumed != text.size()) {
      return false;
    }
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

bool parseSigned(const std::string& text, int* value) {
  try {
    size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed, 0);
    if (consumed != text.size()) {
      return false;
    }
    *value = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

void printUsage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --dataset d1|d3|df          Input generator or file dataset\n"
      << "  --size-bytes N              Bytes in each batch member\n"
      << "  --probbits 9|10|11          rANS probability precision\n"
      << "  --num-in-batch 1|8          Number of independent batch members\n"
      << "  --seed N                    Fixed random seed\n"
      << "  --lambda X                  Exponential lambda for D3\n"
      << "  --file PATH                 Binary input for DF\n"
      << "  --warmup N                  Warmups (must be >= 10)\n"
      << "  --measure N                 Measurements (must be >= 50)\n"
      << "  --gpu-index N               Physical GPU selected by HIP_VISIBLE_DEVICES\n"
      << "  --csv PATH                  CSV file to append\n"
      << "  --case-label TEXT           Human-readable scale label\n"
      << "  --verify-checksum-contract  Run supported checksum combinations\n";
}

bool parseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help") {
      printUsage(argv[0]);
      std::exit(0);
    }
    if (arg == "--verify-checksum-contract") {
      options->verify_checksum_contract = true;
      continue;
    }
    if (i + 1 >= argc) {
      std::cerr << "ERROR: missing value after " << arg << "\n";
      return false;
    }
    const std::string value = argv[++i];
    if (arg == "--dataset") {
      options->dataset = value;
    } else if (arg == "--file") {
      options->file = value;
    } else if (arg == "--csv") {
      options->csv = value;
    } else if (arg == "--case-label") {
      options->case_label = value;
    } else if (arg == "--size-bytes") {
      if (!parseUnsigned(value, &options->size_bytes)) {
        std::cerr << "ERROR: invalid --size-bytes: " << value << "\n";
        return false;
      }
    } else if (arg == "--seed") {
      if (!parseUnsigned(value, &options->seed)) {
        std::cerr << "ERROR: invalid --seed: " << value << "\n";
        return false;
      }
    } else if (arg == "--probbits") {
      if (!parseSigned(value, &options->prob_bits)) {
        std::cerr << "ERROR: invalid --probbits: " << value << "\n";
        return false;
      }
    } else if (arg == "--num-in-batch") {
      uint64_t parsed = 0;
      if (!parseUnsigned(value, &parsed) ||
          parsed > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "ERROR: invalid --num-in-batch: " << value << "\n";
        return false;
      }
      options->num_in_batch = static_cast<uint32_t>(parsed);
    } else if (arg == "--warmup") {
      if (!parseSigned(value, &options->warmup_iters)) {
        std::cerr << "ERROR: invalid --warmup: " << value << "\n";
        return false;
      }
    } else if (arg == "--measure") {
      if (!parseSigned(value, &options->measure_iters)) {
        std::cerr << "ERROR: invalid --measure: " << value << "\n";
        return false;
      }
    } else if (arg == "--gpu-index") {
      if (!parseSigned(value, &options->gpu_index)) {
        std::cerr << "ERROR: invalid --gpu-index: " << value << "\n";
        return false;
      }
    } else if (arg == "--lambda") {
      try {
        size_t consumed = 0;
        options->lambda = std::stod(value, &consumed);
        if (consumed != value.size()) {
          return false;
        }
      } catch (...) {
        std::cerr << "ERROR: invalid --lambda: " << value << "\n";
        return false;
      }
    } else {
      std::cerr << "ERROR: unknown option: " << arg << "\n";
      return false;
    }
  }

  if (options->dataset != "d1" && options->dataset != "d3" &&
      options->dataset != "df") {
    std::cerr << "ERROR: --dataset must be d1, d3, or df\n";
    return false;
  }
  if (options->size_bytes == 0 ||
      options->size_bytes > std::numeric_limits<uint32_t>::max()) {
    std::cerr << "ERROR: --size-bytes must be in [1, UINT32_MAX]\n";
    return false;
  }
  if (options->prob_bits != 9 && options->prob_bits != 10 &&
      options->prob_bits != 11) {
    std::cerr << "ERROR: --probbits must be 9, 10, or 11\n";
    return false;
  }
  if (options->num_in_batch != 1 && options->num_in_batch != 8) {
    std::cerr << "ERROR: --num-in-batch must be 1 or 8 for this harness\n";
    return false;
  }
  if (options->warmup_iters < 10) {
    std::cerr << "ERROR: --warmup must be >= 10\n";
    return false;
  }
  if (options->measure_iters < 50) {
    std::cerr << "ERROR: --measure must be >= 50\n";
    return false;
  }
  if (options->lambda <= 0.0 || !std::isfinite(options->lambda)) {
    std::cerr << "ERROR: --lambda must be finite and positive\n";
    return false;
  }
  if (options->dataset == "df" && options->file.empty()) {
    std::cerr << "ERROR: --file is required for dataset df\n";
    return false;
  }
  if (options->csv.empty()) {
    std::cerr << "ERROR: --csv is required\n";
    return false;
  }
  if (options->gpu_index < 0) {
    std::cerr << "ERROR: --gpu-index must be non-negative\n";
    return false;
  }
  return true;
}

std::string datasetName(const Options& options) {
  if (options.dataset == "d1") {
    return "D1_uniform";
  }
  if (options.dataset == "d3") {
    return "D3_exponential";
  }
  return "DF_file";
}

bool prepareInput(const Options& options, std::vector<uint8_t>* output,
                  std::string* dataset_note) {
  const uint64_t total_u64 =
      options.size_bytes * static_cast<uint64_t>(options.num_in_batch);
  if (total_u64 > std::numeric_limits<size_t>::max()) {
    std::cerr << "ERROR: host input size overflows size_t\n";
    return false;
  }
  output->resize(static_cast<size_t>(total_u64));

  std::mt19937_64 generator(options.seed);
  if (options.dataset == "d1") {
    std::uniform_int_distribution<unsigned int> distribution(0, 255);
    for (uint8_t& value : *output) {
      value = static_cast<uint8_t>(distribution(generator));
    }
    *dataset_note = "uniform random bytes";
    return true;
  }

  if (options.dataset == "d3") {
    std::exponential_distribution<double> distribution(options.lambda);
    for (uint8_t& value : *output) {
      const double sample = std::min(distribution(generator), 1.0);
      const int scaled = std::min(255, static_cast<int>(sample * 256.0));
      value = static_cast<uint8_t>(scaled);
    }
    *dataset_note = "exponential synthetic bytes; lambda=" +
                    fixed(options.lambda, 6);
    return true;
  }

  std::ifstream input(options.file, std::ios::binary);
  if (!input) {
    std::cerr << "ERROR: cannot open DF input: " << options.file << "\n";
    return false;
  }
  input.seekg(0, std::ios::end);
  const std::streamoff end = input.tellg();
  input.seekg(0, std::ios::beg);
  if (end <= 0) {
    std::cerr << "ERROR: DF input is empty: " << options.file << "\n";
    return false;
  }
  std::vector<uint8_t> file_data(static_cast<size_t>(end));
  input.read(reinterpret_cast<char*>(file_data.data()), end);
  if (!input) {
    std::cerr << "ERROR: failed to read DF input: " << options.file << "\n";
    return false;
  }
  for (size_t i = 0; i < output->size(); ++i) {
    (*output)[i] = file_data[i % file_data.size()];
  }
  *dataset_note = "file bytes repeated or truncated to each requested case; "
                  "source=" + options.file + "; source_bytes=" +
                  std::to_string(file_data.size());
  return true;
}

bool rangesOverlap(const void* first, size_t first_size,
                   const void* second, size_t second_size) {
  const uintptr_t first_begin = reinterpret_cast<uintptr_t>(first);
  const uintptr_t second_begin = reinterpret_cast<uintptr_t>(second);
  const uintptr_t first_end = first_begin + first_size;
  const uintptr_t second_end = second_begin + second_size;
  return first_begin < second_end && second_begin < first_end;
}

bool encode(hipans::StackDeviceMemory& stack,
            const hipans::ANSCodecConfig& config,
            std::vector<const void*>& input_ptrs,
            const std::vector<uint32_t>& input_sizes,
            std::vector<void*>& compressed_ptrs,
            uint32_t* compressed_sizes_device,
            hipStream_t stream,
            std::vector<uint32_t>* compressed_sizes_host) {
  hipans::ansEncodeBatchPointer(
      stack, config, static_cast<uint32_t>(input_ptrs.size()),
      input_ptrs.data(), input_sizes.data(), nullptr,
      compressed_ptrs.data(), compressed_sizes_device, stream);
  if (!checkHip(hipStreamSynchronize(stream), "encode hipStreamSynchronize")) {
    return false;
  }
  compressed_sizes_host->resize(input_ptrs.size());
  if (!checkHip(hipMemcpy(compressed_sizes_host->data(),
                          compressed_sizes_device,
                          compressed_sizes_host->size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost),
                "copy compressed sizes D2H")) {
    return false;
  }
  return true;
}

bool decodeCorrectness(hipans::StackDeviceMemory& stack,
                       const hipans::ANSCodecConfig& config,
                       std::vector<const void*>& compressed_ptrs,
                       std::vector<void*>& output_ptrs,
                       const std::vector<uint32_t>& output_capacities,
                       uint8_t* success_device,
                       uint32_t* output_sizes_device,
                       hipStream_t stream,
                       const std::vector<uint8_t>& expected,
                       uint64_t bytes_per_batch,
                       const std::string& label) {
  for (void* output : output_ptrs) {
    if (!checkHip(hipMemsetAsync(output, 0xA5,
                                 static_cast<size_t>(bytes_per_batch), stream),
                  "correctness output prefill")) {
      return false;
    }
  }

  const hipans::ANSDecodeStatus decode_status =
      hipans::ansDecodeBatchPointer(
          stack, config, static_cast<uint32_t>(compressed_ptrs.size()),
          compressed_ptrs.data(), output_ptrs.data(),
          output_capacities.data(), success_device, output_sizes_device,
          stream);
  if (!checkHip(hipStreamSynchronize(stream),
                "correctness decode hipStreamSynchronize")) {
    return false;
  }
  if (decode_status.error != hipans::ANSDecodeError::None) {
    std::cerr << "FAIL " << label << ": ANSDecodeStatus error="
              << static_cast<uint32_t>(decode_status.error) << "\n";
    for (const auto& error : decode_status.errorInfo) {
      std::cerr << "  batch=" << error.first << " " << error.second;
    }
    return false;
  }

  std::vector<uint8_t> success(compressed_ptrs.size());
  std::vector<uint32_t> output_sizes(compressed_ptrs.size());
  if (!checkHip(hipMemcpy(success.data(), success_device, success.size(),
                          hipMemcpyDeviceToHost),
                "copy decode success D2H") ||
      !checkHip(hipMemcpy(output_sizes.data(), output_sizes_device,
                          output_sizes.size() * sizeof(uint32_t),
                          hipMemcpyDeviceToHost),
                "copy decode sizes D2H")) {
    return false;
  }

  std::vector<uint8_t> actual(expected.size());
  for (size_t batch = 0; batch < output_ptrs.size(); ++batch) {
    if (!checkHip(hipMemcpy(actual.data() + batch * bytes_per_batch,
                            output_ptrs[batch], bytes_per_batch,
                            hipMemcpyDeviceToHost),
                  "copy decoded bytes D2H")) {
      return false;
    }
    if (success[batch] == 0) {
      std::cerr << "FAIL " << label << ": batch " << batch
                << " reported decode failure\n";
      return false;
    }
    if (output_sizes[batch] != bytes_per_batch) {
      std::cerr << "FAIL " << label << ": batch " << batch
                << " output size expected=" << bytes_per_batch
                << " actual=" << output_sizes[batch] << "\n";
      return false;
    }
  }

  if (std::memcmp(expected.data(), actual.data(), expected.size()) != 0) {
    size_t offset = 0;
    while (offset < expected.size() && expected[offset] == actual[offset]) {
      ++offset;
    }
    const uint64_t batch = offset / bytes_per_batch;
    const uint64_t in_batch = offset % bytes_per_batch;
    std::cerr << "FAIL " << label << ": first mismatch global_offset="
              << offset << " batch=" << batch << " offset=" << in_batch
              << " expected=" << static_cast<unsigned int>(expected[offset])
              << " actual=" << static_cast<unsigned int>(actual[offset])
              << "\n";
    return false;
  }

  std::cout << "PASS " << label << ": bit-exact " << expected.size()
            << " bytes\n";
  return true;
}

__global__ void outputChecksumKernel(const uint8_t* data, uint64_t size,
                                     unsigned long long* result) {
  __shared__ unsigned long long partial[256];
  const uint64_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
  unsigned long long local = 0;
  for (uint64_t i = thread; i < size; i += stride) {
    local += static_cast<unsigned long long>(data[i]) *
             static_cast<unsigned long long>((i & 255U) + 1U);
  }
  partial[threadIdx.x] = local;
  __syncthreads();
  for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicAdd(result, partial[0]);
  }
}

bool checksumOutputs(const std::vector<void*>& output_ptrs,
                     uint64_t bytes_per_batch,
                     unsigned long long* checksum_device,
                     hipStream_t stream,
                     std::vector<unsigned long long>* checksums) {
  if (!checkHip(hipMemsetAsync(checksum_device, 0,
                               output_ptrs.size() *
                                   sizeof(unsigned long long),
                               stream),
                "clear output checksums")) {
    return false;
  }
  constexpr unsigned int threads = 256;
  const uint64_t needed_blocks =
      (bytes_per_batch + threads - 1) / threads;
  const unsigned int blocks = static_cast<unsigned int>(
      std::max<uint64_t>(1, std::min<uint64_t>(needed_blocks, 4096)));
  for (size_t batch = 0; batch < output_ptrs.size(); ++batch) {
    hipLaunchKernelGGL(outputChecksumKernel, dim3(blocks), dim3(threads), 0,
                       stream,
                       static_cast<const uint8_t*>(output_ptrs[batch]),
                       bytes_per_batch, checksum_device + batch);
    if (!checkHip(hipGetLastError(), "launch output checksum kernel")) {
      return false;
    }
  }
  if (!checkHip(hipStreamSynchronize(stream),
                "output checksum hipStreamSynchronize")) {
    return false;
  }
  checksums->resize(output_ptrs.size());
  return checkHip(hipMemcpy(checksums->data(), checksum_device,
                            checksums->size() *
                                sizeof(unsigned long long),
                            hipMemcpyDeviceToHost),
                  "copy output checksums D2H");
}

int run(int argc, char** argv, ScopedStderrCapture* stderr_capture) {
  Options options;
  if (!parseOptions(argc, argv, &options)) {
    printUsage(argv[0]);
    return 2;
  }

  const char* visible_environment = std::getenv("HIP_VISIBLE_DEVICES");
  const std::string visible =
      visible_environment == nullptr ? "unset" : visible_environment;
  if (visible != std::to_string(options.gpu_index)) {
    std::cerr << "ERROR: HIP_VISIBLE_DEVICES=" << visible
              << " does not match --gpu-index=" << options.gpu_index << "\n";
    return 2;
  }

  if (!checkHip(hipSetDevice(0), "hipSetDevice(logical device 0)")) {
    return 3;
  }
  int device_count = 0;
  if (!checkHip(hipGetDeviceCount(&device_count), "hipGetDeviceCount") ||
      device_count != 1) {
    std::cerr << "ERROR: expected exactly one HIP-visible GPU, found "
              << device_count << "\n";
    return 3;
  }
  hipDeviceProp_t properties{};
  if (!checkHip(hipGetDeviceProperties(&properties, 0),
                "hipGetDeviceProperties")) {
    return 3;
  }

  const std::string host = hostname();
  const std::string rocm_version =
      commandValue("cat /opt/rocm/.info/version", "unknown");
  const std::string hipans_commit =
      commandValue(std::string("git -C ") + kRepo + " rev-parse HEAD",
                   "unknown");
  const std::string harness_commit =
      commandValue(std::string("git -C ") + kBenchDir + " rev-parse HEAD",
                   "NOT_A_GIT_REPOSITORY");
  const uint64_t uncompressed_bytes =
      options.size_bytes * static_cast<uint64_t>(options.num_in_batch);

  std::cout << "============================================================\n";
  std::cout << "hipANS raw ANS decode benchmark\n";
  std::cout << "timestamp: " << isoTimestamp() << "\n";
  std::cout << "hostname: " << host << "\n";
  std::cout << "SLURM_JOB_ID: "
            << (std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID")
                                             : "unset")
            << "\n";
  std::cout << "HIP_VISIBLE_DEVICES: " << visible << "\n";
  std::cout << "physical_gpu_index: " << options.gpu_index << "\n";
  std::cout << "logical_gpu_index: 0\n";
  std::cout << "gpu_name: " << properties.name << "\n";
  std::cout << "gpu_arch: " << properties.gcnArchName << "\n";
  std::cout << "rocm_version: " << rocm_version << "\n";
  std::cout << "hipans_commit: " << hipans_commit << "\n";
  std::cout << "harness_commit: " << harness_commit << "\n";
  std::cout << "build_flags: " << kBuildFlags << "\n";
  std::cout << "case_label: " << options.case_label << "\n";
  std::cout << "dataset: " << datasetName(options) << "\n";
  std::cout << "input_file: "
            << (options.file.empty() ? "N/A" : options.file) << "\n";
  std::cout << "seed: " << options.seed << "\n";
  std::cout << "lambda: " << options.lambda << "\n";
  std::cout << "probbits: " << options.prob_bits << "\n";
  std::cout << "num_in_batch: " << options.num_in_batch << "\n";
  std::cout << "size_bytes_per_batch: " << options.size_bytes << "\n";
  std::cout << "uncompressed_bytes_total: " << uncompressed_bytes << "\n";
  std::cout << "rans_block_bytes: " << kDefaultBlockSize << "\n";
  std::cout << "rans_blocks_per_batch: "
            << ((options.size_bytes + kDefaultBlockSize - 1) /
                kDefaultBlockSize)
            << "\n";
  std::cout << "codec_use_checksum: false (encode and timed decode)\n";
  std::cout << "warmup_iters: " << options.warmup_iters << "\n";
  std::cout << "measure_iters: " << options.measure_iters << "\n";
  std::cout << "stack_device_memory_bytes: " << kStackBytes << "\n";
  std::cout << "csv: " << options.csv << "\n";
  std::cout << "verify_checksum_contract: "
            << (options.verify_checksum_contract ? "true" : "false") << "\n";
  std::cout << "timing_scope: ansDecodeBatchPointer only\n";
  std::cout << "throughput_units: decimal GB/s (1 GB = 1e9 bytes)\n";
  std::cout << "============================================================\n";

  std::vector<uint8_t> host_input;
  std::string dataset_note;
  if (!prepareInput(options, &host_input, &dataset_note)) {
    return 4;
  }
  std::cout << "dataset_details: " << dataset_note << "\n";

  const uint32_t compressed_stride = hipans::getMaxCompressedSize(
      static_cast<uint32_t>(options.size_bytes));
  const uint64_t compressed_capacity_total =
      static_cast<uint64_t>(compressed_stride) * options.num_in_batch;
  std::cout << "compressed_capacity_per_batch: " << compressed_stride << "\n";
  std::cout << "compressed_capacity_total: " << compressed_capacity_total
            << "\n";

  DeviceBuffer<uint8_t> input_device;
  DeviceBuffer<uint8_t> compressed_device;
  DeviceBuffer<uint8_t> output_device;
  DeviceBuffer<uint32_t> compressed_sizes_device;
  DeviceBuffer<uint8_t> output_success_device;
  DeviceBuffer<uint32_t> output_sizes_device;
  DeviceBuffer<unsigned long long> output_checksums_device;

  if (!input_device.allocate(static_cast<size_t>(uncompressed_bytes),
                             "input") ||
      !compressed_device.allocate(
          static_cast<size_t>(compressed_capacity_total), "compressed") ||
      !output_device.allocate(static_cast<size_t>(uncompressed_bytes),
                              "output") ||
      !compressed_sizes_device.allocate(options.num_in_batch,
                                         "compressed sizes") ||
      !output_success_device.allocate(options.num_in_batch,
                                      "decode success") ||
      !output_sizes_device.allocate(options.num_in_batch,
                                    "decode sizes") ||
      !output_checksums_device.allocate(options.num_in_batch,
                                        "output checksums")) {
    return 5;
  }

  if (rangesOverlap(input_device.data(), input_device.bytes(),
                    compressed_device.data(), compressed_device.bytes()) ||
      rangesOverlap(input_device.data(), input_device.bytes(),
                    output_device.data(), output_device.bytes()) ||
      rangesOverlap(compressed_device.data(), compressed_device.bytes(),
                    output_device.data(), output_device.bytes())) {
    std::cerr << "ERROR: caller-owned input/compressed/output buffers overlap\n";
    return 5;
  }
  std::cout << "buffer_aliasing: none; compressed input is immutable and no "
               "measurement uses a prior output as input\n";

  hipStream_t stream = nullptr;
  if (!checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags")) {
    return 5;
  }

  int return_code = 0;
  {
    auto stack = hipans::makeStackMemory(kStackBytes);

    if (!checkHip(hipMemcpyAsync(input_device.data(), host_input.data(),
                                 host_input.size(), hipMemcpyHostToDevice,
                                 stream),
                  "input H2D") ||
        !checkHip(hipStreamSynchronize(stream), "input H2D synchronize")) {
      return_code = 6;
    }

    std::vector<const void*> input_ptrs(options.num_in_batch);
    std::vector<const void*> compressed_ptrs(options.num_in_batch);
    std::vector<void*> compressed_output_ptrs(options.num_in_batch);
    std::vector<void*> output_ptrs(options.num_in_batch);
    std::vector<uint32_t> input_sizes(options.num_in_batch,
                                      static_cast<uint32_t>(options.size_bytes));
    std::vector<uint32_t> output_capacities(
        options.num_in_batch, static_cast<uint32_t>(options.size_bytes));

    for (uint32_t batch = 0; batch < options.num_in_batch; ++batch) {
      input_ptrs[batch] = input_device.data() + batch * options.size_bytes;
      compressed_ptrs[batch] =
          compressed_device.data() + batch * compressed_stride;
      compressed_output_ptrs[batch] =
          compressed_device.data() + batch * compressed_stride;
      output_ptrs[batch] = output_device.data() + batch * options.size_bytes;
    }

    const hipans::ANSCodecConfig raw_config(options.prob_bits, false);
    std::vector<uint32_t> compressed_sizes;
    if (return_code == 0 &&
        !encode(stack, raw_config, input_ptrs, input_sizes,
                compressed_output_ptrs, compressed_sizes_device.data(), stream,
                &compressed_sizes)) {
      return_code = 6;
    }

    uint64_t compressed_bytes = 0;
    if (return_code == 0) {
      for (size_t batch = 0; batch < compressed_sizes.size(); ++batch) {
        if (compressed_sizes[batch] == 0 ||
            compressed_sizes[batch] > compressed_stride) {
          std::cerr << "ERROR: compressed size out of range for batch "
                    << batch << ": " << compressed_sizes[batch]
                    << " capacity=" << compressed_stride << "\n";
          return_code = 6;
          break;
        }
        compressed_bytes += compressed_sizes[batch];
      }
    }
    std::cout << "compressed_bytes_total: " << compressed_bytes << "\n";
    if (uncompressed_bytes != 0) {
      std::cout << "compression_ratio_compressed_over_uncompressed: "
                << fixed(static_cast<double>(compressed_bytes) /
                             static_cast<double>(uncompressed_bytes),
                         9)
                << "\n";
    }

    bool correctness_pass = false;
    if (return_code == 0) {
      correctness_pass = decodeCorrectness(
          stack, raw_config, compressed_ptrs, output_ptrs, output_capacities,
          output_success_device.data(), output_sizes_device.data(), stream,
          host_input, options.size_bytes, "correctness");
      if (!correctness_pass) {
        return_code = 7;
      }
    }

    if (return_code == 0 && options.verify_checksum_contract) {
      std::cout << "----- checksum contract verification (outside timing) -----\n";
      const hipans::ANSCodecConfig checksum_config(options.prob_bits, true);
      std::vector<uint32_t> checksum_compressed_sizes;
      bool contract_ok = encode(
          stack, checksum_config, input_ptrs, input_sizes,
          compressed_output_ptrs, compressed_sizes_device.data(), stream,
          &checksum_compressed_sizes);
      if (contract_ok) {
        contract_ok = decodeCorrectness(
            stack, checksum_config, compressed_ptrs, output_ptrs,
            output_capacities, output_success_device.data(),
            output_sizes_device.data(), stream, host_input, options.size_bytes,
            "checksum=true encode / checksum=true decode");
      }
      if (contract_ok) {
        contract_ok = decodeCorrectness(
            stack, raw_config, compressed_ptrs, output_ptrs,
            output_capacities, output_success_device.data(),
            output_sizes_device.data(), stream, host_input, options.size_bytes,
            "checksum=true encode / checksum=false decode");
      }
      std::cout << "checksum_contract_source_result: checksum-enabled decode "
                   "reads and compares an archive checksum, so it requires a "
                   "checksum-enabled archive; raw checksum=false decode "
                   "does not read that field. Benchmark encode and decode "
                   "therefore both use false.\n";
      if (!contract_ok ||
          !encode(stack, raw_config, input_ptrs, input_sizes,
                  compressed_output_ptrs, compressed_sizes_device.data(),
                  stream, &compressed_sizes)) {
        std::cerr << "FAIL checksum contract verification\n";
        return_code = 8;
      } else {
        std::cout << "PASS checksum contract verification\n";
      }
    }

    std::vector<double> kernel_ms;
    std::vector<double> wall_ms;
    GpuSample precheck;
    GpuSample begin;
    GpuSample middle;
    GpuSample end;
    bool contended = false;
    bool perf_high = false;
    std::string perf_note;

    if (return_code == 0) {
      const std::string perf_command =
          "rocm-smi -d " + std::to_string(options.gpu_index) +
          " --setperflevel high";
      const CommandResult perf_result = runCommand(perf_command);
      std::cout << "----- performance level request -----\n";
      std::cout << "$ " << perf_command << "\n" << perf_result.output;
      perf_high = perf_result.exit_code == 0;
      if (perf_high) {
        perf_note = "setperflevel_high_succeeded";
        std::cout << "frequency_lock: requested performance level high\n";
      } else {
        perf_note = "setperflevel_high_failed";
        std::cout << "frequency_lock: 未锁频\n";
      }

      std::cout << "----- warmup (discarded) -----\n";
      for (int iteration = 0; iteration < options.warmup_iters; ++iteration) {
        const hipans::ANSDecodeStatus status =
            hipans::ansDecodeBatchPointer(
                stack, raw_config, options.num_in_batch, compressed_ptrs.data(),
                output_ptrs.data(), output_capacities.data(),
                output_success_device.data(), output_sizes_device.data(),
                stream);
        if (status.error != hipans::ANSDecodeError::None ||
            !checkHip(hipStreamSynchronize(stream), "warmup synchronize")) {
          std::cerr << "ERROR: warmup decode failed at iteration " << iteration
                    << "\n";
          return_code = 9;
          break;
        }
      }
    }

    if (return_code == 0) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      precheck = sampleGpu("pre-measurement utilization check",
                           options.gpu_index);
      if (!precheck.command_ok || !precheck.utilization ||
          *precheck.utilization != 0.0) {
        contended = true;
        std::cout << "WARNING: target GPU utilization was not confirmed as "
                     "0%; run marked CONTENDED\n";
      } else {
        std::cout << "precheck: target GPU utilization is 0%\n";
      }

      begin = sampleGpu("measurement begin", options.gpu_index);

      hipEvent_t event_start = nullptr;
      hipEvent_t event_stop = nullptr;
      if (!checkHip(hipEventCreate(&event_start), "hipEventCreate(start)") ||
          !checkHip(hipEventCreate(&event_stop), "hipEventCreate(stop)")) {
        return_code = 10;
      } else {
        kernel_ms.reserve(options.measure_iters);
        wall_ms.reserve(options.measure_iters);
        std::cout << "----- measurement: decode only -----\n";
        for (int iteration = 0; iteration < options.measure_iters; ++iteration) {
          if (!checkHip(hipEventRecord(event_start, stream),
                        "hipEventRecord(start)")) {
            return_code = 10;
            break;
          }
          const auto wall_start = std::chrono::steady_clock::now();
          const hipans::ANSDecodeStatus status =
              hipans::ansDecodeBatchPointer(
                  stack, raw_config, options.num_in_batch,
                  compressed_ptrs.data(), output_ptrs.data(),
                  output_capacities.data(), output_success_device.data(),
                  output_sizes_device.data(), stream);
          if (!checkHip(hipEventRecord(event_stop, stream),
                        "hipEventRecord(stop)") ||
              !checkHip(hipEventSynchronize(event_stop),
                        "hipEventSynchronize(stop)")) {
            return_code = 10;
            break;
          }
          const auto wall_stop = std::chrono::steady_clock::now();
          if (status.error != hipans::ANSDecodeError::None) {
            std::cerr << "ERROR: timed decode returned error at iteration "
                      << iteration << "\n";
            return_code = 10;
            break;
          }
          float elapsed = 0.0f;
          if (!checkHip(hipEventElapsedTime(&elapsed, event_start, event_stop),
                        "hipEventElapsedTime")) {
            return_code = 10;
            break;
          }
          const double wall_elapsed =
              std::chrono::duration<double, std::milli>(wall_stop - wall_start)
                  .count();
          kernel_ms.push_back(static_cast<double>(elapsed));
          wall_ms.push_back(wall_elapsed);

          if (iteration + 1 == options.measure_iters / 2) {
            middle = sampleGpu("measurement midpoint", options.gpu_index);
          }
        }
        if (!checkHip(hipEventDestroy(event_start),
                      "hipEventDestroy(start)") && return_code == 0) {
          return_code = 10;
        }
        if (!checkHip(hipEventDestroy(event_stop),
                      "hipEventDestroy(stop)") && return_code == 0) {
          return_code = 10;
        }
      }
      end = sampleGpu("measurement end", options.gpu_index);
    }

    std::vector<unsigned long long> output_checksums;
    if (return_code == 0 &&
        !checksumOutputs(output_ptrs, options.size_bytes,
                         output_checksums_device.data(), stream,
                         &output_checksums)) {
      return_code = 11;
    }
    unsigned long long checksum_fold = 0;
    for (size_t batch = 0; batch < output_checksums.size(); ++batch) {
      checksum_fold ^= output_checksums[batch] +
                       0x9e3779b97f4a7c15ULL + (checksum_fold << 6) +
                       (checksum_fold >> 2);
    }
    if (!output_checksums.empty()) {
      std::cout << "post_measurement_output_checksum: 0x" << std::hex
                << checksum_fold << std::dec
                << " (reads every output byte outside timing)\n";
    }

    std::vector<uint8_t> final_success(options.num_in_batch);
    std::vector<uint32_t> final_sizes(options.num_in_batch);
    if (return_code == 0 &&
        (!checkHip(hipMemcpy(final_success.data(),
                             output_success_device.data(),
                             final_success.size(), hipMemcpyDeviceToHost),
                   "copy final success") ||
         !checkHip(hipMemcpy(final_sizes.data(), output_sizes_device.data(),
                             final_sizes.size() * sizeof(uint32_t),
                             hipMemcpyDeviceToHost),
                   "copy final sizes"))) {
      return_code = 11;
    }
    if (return_code == 0) {
      for (uint32_t batch = 0; batch < options.num_in_batch; ++batch) {
        if (final_success[batch] == 0 ||
            final_sizes[batch] != options.size_bytes) {
          std::cerr << "ERROR: final timed output status failed for batch "
                    << batch << " success="
                    << static_cast<unsigned int>(final_success[batch])
                    << " size=" << final_sizes[batch] << "\n";
          return_code = 11;
          break;
        }
      }
    }

    const bool overflow_warning =
        stderr_capture->captured().find(kOverflowWarning) != std::string::npos;
    std::string run_status = contended ? "CONTENDED" : "OK";
    if (overflow_warning) {
      run_status = "INVALID";
      std::cerr << "ERROR: detected StackDeviceMemory overflow warning; run "
                   "marked INVALID\n";
      if (return_code == 0) {
        return_code = 12;
      }
    }
    if (return_code != 0 && run_status != "INVALID") {
      run_status = "INVALID";
    }

    Stats kernel_stats;
    Stats wall_stats;
    if (!kernel_ms.empty()) {
      kernel_stats = calculateStats(kernel_ms);
      wall_stats = calculateStats(wall_ms);
    }
    const double compression_ratio =
        uncompressed_bytes == 0
            ? 0.0
            : static_cast<double>(compressed_bytes) /
                  static_cast<double>(uncompressed_bytes);
    const double seconds = kernel_stats.median / 1000.0;
    const double throughput_uncompressed =
        seconds > 0.0
            ? static_cast<double>(uncompressed_bytes) / seconds / 1.0e9
            : 0.0;
    const double throughput_compressed =
        seconds > 0.0
            ? static_cast<double>(compressed_bytes) / seconds / 1.0e9
            : 0.0;
    const double throughput_total = throughput_uncompressed +
                                    throughput_compressed;

    std::cout << "----- result summary -----\n";
    std::cout << "correctness: "
              << (correctness_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "run_status: " << run_status << "\n";
    std::cout << "stack_max_memory_usage_bytes: "
              << stack.getMaxMemoryUsage() << "\n";
    std::cout << "kernel_ms median=" << fixed(kernel_stats.median)
              << " p5=" << fixed(kernel_stats.p5)
              << " p95=" << fixed(kernel_stats.p95)
              << " min=" << fixed(kernel_stats.min)
              << " max=" << fixed(kernel_stats.max) << "\n";
    std::cout << "wall_ms median=" << fixed(wall_stats.median)
              << " p5=" << fixed(wall_stats.p5)
              << " p95=" << fixed(wall_stats.p95)
              << " min=" << fixed(wall_stats.min)
              << " max=" << fixed(wall_stats.max) << "\n";
    std::cout << "throughput_GBps_uncompressed: "
              << fixed(throughput_uncompressed) << "\n";
    std::cout << "throughput_GBps_compressed: "
              << fixed(throughput_compressed) << "\n";
    std::cout << "throughput_GBps_total_traffic: "
              << fixed(throughput_total) << "\n";

    std::ostringstream notes;
    notes << "case=" << options.case_label << "; " << dataset_note
          << "; checksum_encode=false; checksum_decode=false; "
          << perf_note << "; mclk_begin=" << optionalNumber(begin.mclk_mhz)
          << "; mclk_mid=" << optionalNumber(middle.mclk_mhz)
          << "; mclk_end=" << optionalNumber(end.mclk_mhz)
          << "; temp_mid=" << optionalNumber(middle.temperature_c)
          << "; output_checksum=0x" << std::hex << checksum_fold << std::dec;

    const std::string metric_or_blank =
        kernel_ms.empty() ? "" : fixed(kernel_stats.median);
    const std::vector<std::string> csv_values = {
        isoTimestamp(),
        host,
        std::to_string(options.gpu_index),
        properties.gcnArchName,
        rocm_version,
        hipans_commit,
        harness_commit,
        kBuildFlags,
        datasetName(options),
        std::to_string(options.prob_bits),
        std::to_string(options.num_in_batch),
        std::to_string(options.size_bytes),
        std::to_string(uncompressed_bytes),
        std::to_string(compressed_bytes),
        fixed(compression_ratio, 9),
        correctness_pass ? "PASS" : "FAIL",
        run_status,
        std::to_string(options.warmup_iters),
        std::to_string(options.measure_iters),
        metric_or_blank,
        kernel_ms.empty() ? "" : fixed(kernel_stats.p5),
        kernel_ms.empty() ? "" : fixed(kernel_stats.p95),
        kernel_ms.empty() ? "" : fixed(kernel_stats.min),
        kernel_ms.empty() ? "" : fixed(kernel_stats.max),
        wall_ms.empty() ? "" : fixed(wall_stats.median),
        kernel_ms.empty() ? "" : fixed(throughput_uncompressed),
        kernel_ms.empty() ? "" : fixed(throughput_compressed),
        kernel_ms.empty() ? "" : fixed(throughput_total),
        optionalNumber(begin.sclk_mhz),
        optionalNumber(middle.sclk_mhz),
        optionalNumber(end.sclk_mhz),
        optionalNumber(begin.temperature_c),
        optionalNumber(end.temperature_c),
        notes.str()};
    if (!appendCsv(options.csv, csv_values) && return_code == 0) {
      return_code = 13;
    }
    std::cout << "csv_row_appended: " << options.csv << "\n";

    if (perf_high) {
      const std::string restore_command =
          "rocm-smi -d " + std::to_string(options.gpu_index) +
          " --setperflevel auto";
      const CommandResult restore = runCommand(restore_command);
      std::cout << "----- performance level restore -----\n";
      std::cout << "$ " << restore_command << "\n" << restore.output;
      std::cout << "restore_rc: " << restore.exit_code << "\n";
    }
  }

  if (!checkHip(hipStreamDestroy(stream), "hipStreamDestroy") &&
      return_code == 0) {
    return_code = 14;
  }
  std::cout << "exit_code: " << return_code << "\n";
  return return_code;
}

}  // namespace

int main(int argc, char** argv) {
  ScopedStderrCapture stderr_capture;
  return run(argc, argv, &stderr_capture);
}
