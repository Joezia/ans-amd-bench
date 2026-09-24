#include <glog/logging.h>

#include "hipans/ans/GpuANSCodec.h"
#include "hipans/utils/StackDeviceMemory.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>

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
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t kStackBytes = 2ULL * 1024 * 1024 * 1024;
constexpr size_t kFlushBytes = 128ULL * 1024 * 1024;
constexpr size_t kLtWorkspaceBytes = 64ULL * 1024 * 1024;
constexpr const char* kOverflowWarning =
    "StackDeviceMemory: attempting to allocate";
#ifndef HIPANS_SOURCE_DIR
#define HIPANS_SOURCE_DIR "unknown"
#endif
#ifndef BENCH_SOURCE_DIR
#define BENCH_SOURCE_DIR "unknown"
#endif
constexpr const char* kRepo = HIPANS_SOURCE_DIR;
constexpr const char* kBenchDir = BENCH_SOURCE_DIR;
constexpr const char* kBuildFlags =
    "-std=c++17 -O3 -DNDEBUG --offload-arch=gfx90a";

struct Options {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  std::string weight_file;
  std::string csv;
  uint64_t seed = 20260921;
  int warmup = 10;
  int measure = 50;
  int gpu_index = 0;
  double cpu_rel_tolerance = 2.0e-2;
};

struct CommandResult {
  int exit_code = -1;
  std::string output;
};

struct Stats {
  double median = 0.0;
  double p5 = 0.0;
  double p95 = 0.0;
  double min = 0.0;
  double max = 0.0;
};

struct GpuSample {
  std::string label;
  std::string raw;
  std::optional<double> utilization;
  std::optional<double> sclk_mhz;
  std::optional<double> mclk_mhz;
  std::optional<double> temperature_c;
  bool ok = false;
};

struct TimingResult {
  Stats gemm;
  Stats decode;
  Stats decode_gemm;
  double ratio = 0.0;
};

class TeeStreamBuf final : public std::streambuf {
 public:
  explicit TeeStreamBuf(std::streambuf* original) : original_(original) {}
  std::string captured() const { return captured_.str(); }

 protected:
  int overflow(int ch) override {
    if (ch == traits_type::eof()) {
      return traits_type::not_eof(ch);
    }
    const char value = static_cast<char>(ch);
    if (original_->sputc(value) == traits_type::eof()) {
      return traits_type::eof();
    }
    captured_.put(value);
    return ch;
  }
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    const std::streamsize written = original_->sputn(data, count);
    captured_.write(data, written);
    return written;
  }
  int sync() override {
    captured_.flush();
    return original_->pubsync();
  }

 private:
  std::streambuf* original_;
  std::ostringstream captured_;
};

class ScopedStderrCapture final {
 public:
  ScopedStderrCapture() : original_(std::cerr.rdbuf()), tee_(original_) {
    std::cerr.rdbuf(&tee_);
  }
  ~ScopedStderrCapture() {
    std::cerr.flush();
    std::cerr.rdbuf(original_);
  }
  std::string captured() const { return tee_.captured(); }

 private:
  std::streambuf* original_;
  TeeStreamBuf tee_;
};

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      const hipError_t status = hipFree(pointer_);
      if (status != hipSuccess) {
        std::cerr << "ERROR: hipFree(" << name_ << "): "
                  << hipGetErrorString(status) << "\n";
      }
    }
  }
  bool allocate(size_t count, const std::string& name) {
    name_ = name;
    count_ = count;
    if (count == 0) {
      std::cerr << "ERROR: zero allocation for " << name << "\n";
      return false;
    }
    const hipError_t status =
        hipMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T));
    if (status != hipSuccess) {
      std::cerr << "ERROR: hipMalloc(" << name << ", "
                << count * sizeof(T) << " bytes): "
                << hipGetErrorString(status) << "\n";
      pointer_ = nullptr;
      count_ = 0;
      return false;
    }
    return true;
  }
  T* data() { return pointer_; }
  const T* data() const { return pointer_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T* pointer_ = nullptr;
  size_t count_ = 0;
  std::string name_;
};

bool checkHip(hipError_t status, const char* operation) {
  if (status == hipSuccess) {
    return true;
  }
  std::cerr << "ERROR: " << operation << ": " << hipGetErrorString(status)
            << "\n";
  return false;
}

bool checkLt(hipblasStatus_t status, const char* operation) {
  if (status == HIPBLAS_STATUS_SUCCESS) {
    return true;
  }
  std::cerr << "ERROR: " << operation << ": hipBLASLt status "
            << static_cast<int>(status) << "\n";
  return false;
}

bool checkRocblas(rocblas_status status, const char* operation) {
  if (status == rocblas_status_success) {
    return true;
  }
  std::cerr << "ERROR: " << operation << ": rocBLAS status "
            << static_cast<int>(status) << "\n";
  return false;
}

std::string trim(std::string value) {
  const size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

CommandResult runCommand(const std::string& command) {
  CommandResult result;
  FILE* pipe = popen((command + " 2>&1").c_str(), "r");
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
  return result.exit_code == 0 && !value.empty() ? value : fallback;
}

std::optional<double> parseLineNumber(const std::string& text,
                                      const std::string& required,
                                      const std::string& pattern) {
  std::istringstream lines(text);
  std::string line;
  const std::regex regex(pattern, std::regex::icase);
  while (std::getline(lines, line)) {
    std::string lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (lower.find(required) == std::string::npos) {
      continue;
    }
    std::smatch match;
    if (std::regex_search(line, match, regex)) {
      return std::stod(match[1].str());
    }
  }
  return std::nullopt;
}

GpuSample sampleGpu(const std::string& label, int gpu_index) {
  GpuSample sample;
  sample.label = label;
  const std::string command = "rocm-smi -d " + std::to_string(gpu_index) +
                              " --showuse --showclocks --showtemp";
  const CommandResult result = runCommand(command);
  sample.raw = result.output;
  sample.ok = result.exit_code == 0;
  sample.utilization = parseLineNumber(result.output, "gpu use",
                                       "GPU use \\(%\\):\\s*([0-9.]+)");
  sample.sclk_mhz = parseLineNumber(result.output, "sclk clock level",
                                    "\\(([0-9.]+)Mhz\\)");
  sample.mclk_mhz = parseLineNumber(result.output, "mclk clock level",
                                    "\\(([0-9.]+)Mhz\\)");
  sample.temperature_c = parseLineNumber(
      result.output, "temperature (sensor edge)", "\\(C\\):\\s*([0-9.]+)");
  std::cout << "----- rocm-smi sample: " << label << " -----\n";
  std::cout << "$ " << command << "\n" << result.output;
  return sample;
}

std::string optionalNumber(const std::optional<double>& value) {
  if (!value) {
    return "NA";
  }
  std::ostringstream out;
  out << std::fixed << std::setprecision(3) << *value;
  return out.str();
}

uint16_t floatToBf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<uint16_t>((bits + rounding) >> 16U);
}

float bf16ToFloat(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float output = 0.0f;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

bool parseInteger(const char* value, int64_t* output) {
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    return false;
  }
  *output = static_cast<int64_t>(parsed);
  return true;
}

bool parseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (i + 1 >= argc) {
      std::cerr << "ERROR: option needs a value: " << argument << "\n";
      return false;
    }
    const char* value = argv[++i];
    int64_t integer = 0;
    if (argument == "--m" && parseInteger(value, &integer)) {
      options->m = integer;
    } else if (argument == "--n" && parseInteger(value, &integer)) {
      options->n = integer;
    } else if (argument == "--k" && parseInteger(value, &integer)) {
      options->k = integer;
    } else if (argument == "--weight-file") {
      options->weight_file = value;
    } else if (argument == "--csv") {
      options->csv = value;
    } else if (argument == "--seed" && parseInteger(value, &integer)) {
      options->seed = static_cast<uint64_t>(integer);
    } else if (argument == "--warmup" && parseInteger(value, &integer)) {
      options->warmup = static_cast<int>(integer);
    } else if (argument == "--measure" && parseInteger(value, &integer)) {
      options->measure = static_cast<int>(integer);
    } else if (argument == "--gpu-index" && parseInteger(value, &integer)) {
      options->gpu_index = static_cast<int>(integer);
    } else if (argument == "--cpu-rel-tolerance") {
      char* end = nullptr;
      options->cpu_rel_tolerance = std::strtod(value, &end);
      if (end == value || *end != '\0') {
        std::cerr << "ERROR: invalid --cpu-rel-tolerance\n";
        return false;
      }
    } else {
      std::cerr << "ERROR: unknown or invalid option: " << argument << "\n";
      return false;
    }
  }
  if (options->m <= 0 || options->n <= 0 || options->k <= 0 ||
      options->weight_file.empty() || options->csv.empty()) {
    std::cerr << "ERROR: --m --n --k --weight-file and --csv are required\n";
    return false;
  }
  if (options->warmup < 10 || options->measure < 50) {
    std::cerr << "ERROR: warmup must be >=10 and measure must be >=50\n";
    return false;
  }
  return true;
}

bool checkedElements(int64_t first, int64_t second, size_t* elements) {
  if (first <= 0 || second <= 0) {
    return false;
  }
  const uint64_t product = static_cast<uint64_t>(first) *
                           static_cast<uint64_t>(second);
  if (product > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *elements = static_cast<size_t>(product);
  return true;
}

bool readWeight(const Options& options, size_t requested_bytes,
                std::vector<uint8_t>* output, uint64_t* source_bytes,
                std::string* method) {
  struct stat info {};
  if (stat(options.weight_file.c_str(), &info) != 0 || info.st_size <= 0) {
    std::cerr << "ERROR: cannot stat non-empty weight file: "
              << options.weight_file << "\n";
    return false;
  }
  *source_bytes = static_cast<uint64_t>(info.st_size);
  const size_t chunk_size = static_cast<size_t>(
      std::min<uint64_t>(*source_bytes, requested_bytes));
  std::vector<uint8_t> chunk(chunk_size);
  std::ifstream input(options.weight_file, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(chunk.data()), chunk.size())) {
    std::cerr << "ERROR: failed reading " << chunk.size() << " bytes from "
              << options.weight_file << "\n";
    return false;
  }
  output->resize(requested_bytes);
  for (size_t offset = 0; offset < requested_bytes; offset += chunk.size()) {
    const size_t count = std::min(chunk.size(), requested_bytes - offset);
    std::memcpy(output->data() + offset, chunk.data(), count);
  }
  if (*source_bytes >= requested_bytes) {
    *method = "prefix_truncated_to_" + std::to_string(requested_bytes) +
              "_bytes";
  } else {
    *method = "full_file_cyclically_repeated_to_" +
              std::to_string(requested_bytes) + "_bytes";
  }
  return true;
}

std::vector<uint16_t> makeActivation(size_t elements, uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::normal_distribution<float> distribution(0.0f, 0.5f);
  std::vector<uint16_t> values(elements);
  for (uint16_t& value : values) {
    value = floatToBf16(distribution(generator));
  }
  return values;
}

Stats calculateStats(std::vector<double> values) {
  Stats stats;
  if (values.empty()) {
    return stats;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&values](double fraction) {
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
  };
  stats.median = percentile(0.5);
  stats.p5 = percentile(0.05);
  stats.p95 = percentile(0.95);
  stats.min = values.front();
  stats.max = values.back();
  return stats;
}

__global__ void flushKernel(uint32_t* data, size_t words, uint32_t salt) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index = thread; index < words; index += stride) {
    data[index] = static_cast<uint32_t>(index) ^ salt;
  }
}

class VendorGemm final {
 public:
  VendorGemm() = default;
  VendorGemm(const VendorGemm&) = delete;
  VendorGemm& operator=(const VendorGemm&) = delete;
  ~VendorGemm() { cleanup(); }

  bool initialize(int64_t m, int64_t n, int64_t k) {
    m_ = m;
    n_ = n;
    k_ = k;
    if (initializeLt()) {
      backend_ = "hipBLASLt";
      int version = 0;
      if (hipblasLtGetVersion(lt_handle_, &version) ==
          HIPBLAS_STATUS_SUCCESS) {
        version_ = std::to_string(version);
      } else {
        version_ = "unknown";
      }
      return true;
    }

    std::cerr << "WARNING: hipBLASLt initialization/heuristic unavailable; "
                 "falling back to rocBLAS rocblas_gemm_ex\n";
    cleanupLt();
    if (!checkRocblas(rocblas_create_handle(&rocblas_handle_),
                      "rocblas_create_handle") ||
        !checkRocblas(rocblas_set_pointer_mode(
                          rocblas_handle_, rocblas_pointer_mode_host),
                      "rocblas_set_pointer_mode")) {
      return false;
    }
    backend_ = "rocBLAS";
    char version[128] = {};
    if (rocblas_get_version_string(version, sizeof(version)) ==
        rocblas_status_success) {
      version_ = version;
    } else {
      version_ = "unknown";
    }
    algorithm_ =
        "rocblas_gemm_ex/rocblas_gemm_algo_standard(enum=0)";
    return true;
  }

  bool run(const void* a, const void* b, void* d, hipStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (backend_ == "hipBLASLt") {
      return checkLt(
          hipblasLtMatmul(lt_handle_, operation_desc_, &alpha, a, a_desc_, b,
                          b_desc_, &beta, d, c_desc_, d, d_desc_, &algorithm_lt_,
                          workspace_.data(), workspace_bytes_, stream),
          "hipblasLtMatmul");
    }
    if (!checkRocblas(rocblas_set_stream(rocblas_handle_, stream),
                      "rocblas_set_stream")) {
      return false;
    }
    // Row-major C=A*B is column-major C^T=B^T*A^T using the same bytes.
    return checkRocblas(
        rocblas_gemm_ex(rocblas_handle_, rocblas_operation_none,
                        rocblas_operation_none, n_, m_, k_, &alpha, b,
                        rocblas_datatype_bf16_r, n_, a,
                        rocblas_datatype_bf16_r, k_, &beta, d,
                        rocblas_datatype_f32_r, n_, d,
                        rocblas_datatype_f32_r, n_, rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard, 0, 0),
        "rocblas_gemm_ex");
  }

  const std::string& backend() const { return backend_; }
  const std::string& version() const { return version_; }
  const std::string& algorithm() const { return algorithm_; }
  size_t workspaceBytes() const { return workspace_bytes_; }

 private:
  bool initializeLt() {
    if (!checkLt(hipblasLtCreate(&lt_handle_), "hipblasLtCreate")) {
      return false;
    }
    if (!checkLt(hipblasLtMatmulDescCreate(&operation_desc_,
                                           HIPBLAS_COMPUTE_32F, HIP_R_32F),
                 "hipblasLtMatmulDescCreate") ||
        !checkLt(hipblasLtMatrixLayoutCreate(&a_desc_, HIP_R_16BF,
                                             static_cast<uint64_t>(m_),
                                             static_cast<uint64_t>(k_), k_),
                 "hipblasLtMatrixLayoutCreate(A)") ||
        !checkLt(hipblasLtMatrixLayoutCreate(&b_desc_, HIP_R_16BF,
                                             static_cast<uint64_t>(k_),
                                             static_cast<uint64_t>(n_), n_),
                 "hipblasLtMatrixLayoutCreate(B)") ||
        !checkLt(hipblasLtMatrixLayoutCreate(&c_desc_, HIP_R_32F,
                                             static_cast<uint64_t>(m_),
                                             static_cast<uint64_t>(n_), n_),
                 "hipblasLtMatrixLayoutCreate(C)") ||
        !checkLt(hipblasLtMatrixLayoutCreate(&d_desc_, HIP_R_32F,
                                             static_cast<uint64_t>(m_),
                                             static_cast<uint64_t>(n_), n_),
                 "hipblasLtMatrixLayoutCreate(D)")) {
      return false;
    }
    const hipblasLtOrder_t row_order = HIPBLASLT_ORDER_ROW;
    if (!checkLt(hipblasLtMatrixLayoutSetAttribute(
                     a_desc_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                     sizeof(row_order)),
                 "set A row order") ||
        !checkLt(hipblasLtMatrixLayoutSetAttribute(
                     b_desc_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                     sizeof(row_order)),
                 "set B row order") ||
        !checkLt(hipblasLtMatrixLayoutSetAttribute(
                     c_desc_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                     sizeof(row_order)),
                 "set C row order") ||
        !checkLt(hipblasLtMatrixLayoutSetAttribute(
                     d_desc_, HIPBLASLT_MATRIX_LAYOUT_ORDER, &row_order,
                     sizeof(row_order)),
                 "set D row order") ||
        !checkLt(hipblasLtMatmulPreferenceCreate(&preference_),
                 "hipblasLtMatmulPreferenceCreate") ||
        !checkLt(hipblasLtMatmulPreferenceSetAttribute(
                     preference_, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                     &kLtWorkspaceBytes, sizeof(kLtWorkspaceBytes)),
                 "set hipBLASLt max workspace")) {
      return false;
    }

    hipblasLtMatmulHeuristicResult_t results[32] = {};
    int returned = 0;
    if (!checkLt(hipblasLtMatmulAlgoGetHeuristic(
                     lt_handle_, operation_desc_, a_desc_, b_desc_, c_desc_,
                     d_desc_, preference_, 32, results, &returned),
                 "hipblasLtMatmulAlgoGetHeuristic") ||
        returned <= 0) {
      std::cerr << "WARNING: hipBLASLt returned no algorithms\n";
      return false;
    }
    int selected = -1;
    for (int index = 0; index < returned; ++index) {
      if (results[index].state == HIPBLAS_STATUS_SUCCESS &&
          results[index].workspaceSize <= kLtWorkspaceBytes) {
        selected = index;
        break;
      }
    }
    if (selected < 0) {
      std::cerr << "WARNING: hipBLASLt returned no usable algorithm\n";
      return false;
    }
    algorithm_lt_ = results[selected].algo;
    workspace_bytes_ = results[selected].workspaceSize;
    if (workspace_bytes_ > 0 &&
        !workspace_.allocate(workspace_bytes_, "hipBLASLt workspace")) {
      return false;
    }
    int algorithm_index = -1;
    std::string solution_name;
    std::string kernel_name;
    try {
      algorithm_index = hipblaslt_ext::getIndexFromAlgo(algorithm_lt_);
      solution_name =
          hipblaslt_ext::getSolutionNameFromAlgo(lt_handle_, algorithm_lt_);
      kernel_name =
          hipblaslt_ext::getKernelNameFromAlgo(lt_handle_, algorithm_lt_);
    } catch (...) {
      solution_name = "extension_lookup_failed";
    }
    std::ostringstream description;
    description << "heuristic_rank=" << selected << ";algo_index="
                << algorithm_index << ";solution=" << solution_name
                << ";kernel=" << kernel_name << ";waves="
                << results[selected].wavesCount;
    algorithm_ = description.str();
    return true;
  }

  void cleanupLt() {
    if (preference_ != nullptr) {
      hipblasLtMatmulPreferenceDestroy(preference_);
      preference_ = nullptr;
    }
    if (d_desc_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(d_desc_);
      d_desc_ = nullptr;
    }
    if (c_desc_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(c_desc_);
      c_desc_ = nullptr;
    }
    if (b_desc_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(b_desc_);
      b_desc_ = nullptr;
    }
    if (a_desc_ != nullptr) {
      hipblasLtMatrixLayoutDestroy(a_desc_);
      a_desc_ = nullptr;
    }
    if (operation_desc_ != nullptr) {
      hipblasLtMatmulDescDestroy(operation_desc_);
      operation_desc_ = nullptr;
    }
    if (lt_handle_ != nullptr) {
      hipblasLtDestroy(lt_handle_);
      lt_handle_ = nullptr;
    }
  }

  void cleanup() {
    cleanupLt();
    if (rocblas_handle_ != nullptr) {
      rocblas_destroy_handle(rocblas_handle_);
      rocblas_handle_ = nullptr;
    }
  }

  int64_t m_ = 0;
  int64_t n_ = 0;
  int64_t k_ = 0;
  std::string backend_;
  std::string version_;
  std::string algorithm_;
  hipblasLtHandle_t lt_handle_ = nullptr;
  hipblasLtMatmulDesc_t operation_desc_ = nullptr;
  hipblasLtMatrixLayout_t a_desc_ = nullptr;
  hipblasLtMatrixLayout_t b_desc_ = nullptr;
  hipblasLtMatrixLayout_t c_desc_ = nullptr;
  hipblasLtMatrixLayout_t d_desc_ = nullptr;
  hipblasLtMatmulPreference_t preference_ = nullptr;
  hipblasLtMatmulAlgo_t algorithm_lt_ = {};
  DeviceBuffer<uint8_t> workspace_;
  size_t workspace_bytes_ = 0;
  rocblas_handle rocblas_handle_ = nullptr;
};

bool encodeWeight(hipans::StackDeviceMemory& stack,
                  const hipans::ANSCodecConfig& config,
                  const void* raw_device, uint32_t raw_bytes,
                  void* compressed_device, uint32_t* compressed_size_device,
                  hipStream_t stream, uint32_t* compressed_size) {
  const void* inputs[1] = {raw_device};
  const uint32_t input_sizes[1] = {raw_bytes};
  void* outputs[1] = {compressed_device};
  hipans::ansEncodeBatchPointer(stack, config, 1, inputs, input_sizes, nullptr,
                                outputs, compressed_size_device, stream);
  if (!checkHip(hipStreamSynchronize(stream), "encode synchronize") ||
      !checkHip(hipMemcpy(compressed_size, compressed_size_device,
                          sizeof(*compressed_size), hipMemcpyDeviceToHost),
                "compressed size D2H")) {
    return false;
  }
  return true;
}

bool enqueueDecode(hipans::StackDeviceMemory& stack,
                   const hipans::ANSCodecConfig& config,
                   const void* compressed_device, void* decoded_device,
                   uint32_t raw_bytes, uint8_t* success_device,
                   uint32_t* size_device, hipStream_t stream) {
  const void* inputs[1] = {compressed_device};
  void* outputs[1] = {decoded_device};
  const uint32_t capacities[1] = {raw_bytes};
  const hipans::ANSDecodeStatus status = hipans::ansDecodeBatchPointer(
      stack, config, 1, inputs, outputs, capacities, success_device,
      size_device, stream);
  if (status.error != hipans::ANSDecodeError::None) {
    std::cerr << "ERROR: ansDecodeBatchPointer host status="
              << static_cast<uint32_t>(status.error) << "\n";
    return false;
  }
  return true;
}

bool verifyDecode(hipans::StackDeviceMemory& stack,
                  const hipans::ANSCodecConfig& config,
                  const std::vector<uint8_t>& expected,
                  const void* compressed_device, void* decoded_device,
                  uint8_t* success_device, uint32_t* size_device,
                  hipStream_t stream) {
  if (!checkHip(hipMemsetAsync(decoded_device, 0xa5, expected.size(), stream),
                "prefill decoded B") ||
      !enqueueDecode(stack, config, compressed_device, decoded_device,
                     static_cast<uint32_t>(expected.size()), success_device,
                     size_device, stream) ||
      !checkHip(hipStreamSynchronize(stream), "correctness decode synchronize")) {
    return false;
  }
  uint8_t success = 0;
  uint32_t size = 0;
  std::vector<uint8_t> actual(expected.size());
  if (!checkHip(hipMemcpy(&success, success_device, sizeof(success),
                          hipMemcpyDeviceToHost),
                "decode success D2H") ||
      !checkHip(hipMemcpy(&size, size_device, sizeof(size),
                          hipMemcpyDeviceToHost),
                "decode size D2H") ||
      !checkHip(hipMemcpy(actual.data(), decoded_device, actual.size(),
                          hipMemcpyDeviceToHost),
                "decoded B D2H")) {
    return false;
  }
  if (success == 0 || size != expected.size()) {
    std::cerr << "FAIL decoded B status: success="
              << static_cast<unsigned>(success) << " size=" << size
              << " expected_size=" << expected.size() << "\n";
    return false;
  }
  if (std::memcmp(expected.data(), actual.data(), expected.size()) != 0) {
    size_t index = 0;
    while (index < expected.size() && expected[index] == actual[index]) {
      ++index;
    }
    std::cerr << "FAIL decoded B byte comparison: offset=" << index
              << " expected=" << static_cast<unsigned>(expected[index])
              << " actual=" << static_cast<unsigned>(actual[index]) << "\n";
    return false;
  }
  std::cout << "PASS decoded B bit-exact: " << expected.size() << " bytes\n";
  return true;
}

bool verifyGemm(VendorGemm& gemm, const Options& options,
                const std::vector<uint16_t>& activation,
                const std::vector<uint8_t>& weight_bytes,
                const uint16_t* activation_device, const void* weight_device,
                const void* decoded_device, float* raw_output_device,
                float* decoded_output_device, hipStream_t stream,
                double* max_abs, double* max_rel, double* cpu_max_rel) {
  if (!gemm.run(activation_device, weight_device, raw_output_device, stream) ||
      !gemm.run(activation_device, decoded_device, decoded_output_device,
                stream) ||
      !checkHip(hipStreamSynchronize(stream), "correctness GEMM synchronize")) {
    return false;
  }
  size_t output_elements = 0;
  if (!checkedElements(options.m, options.n, &output_elements)) {
    return false;
  }
  std::vector<float> raw_output(output_elements);
  std::vector<float> decoded_output(output_elements);
  if (!checkHip(hipMemcpy(raw_output.data(), raw_output_device,
                          raw_output.size() * sizeof(float),
                          hipMemcpyDeviceToHost),
                "raw GEMM output D2H") ||
      !checkHip(hipMemcpy(decoded_output.data(), decoded_output_device,
                          decoded_output.size() * sizeof(float),
                          hipMemcpyDeviceToHost),
                "decoded GEMM output D2H")) {
    return false;
  }

  *max_abs = 0.0;
  *max_rel = 0.0;
  size_t first_mismatch = output_elements;
  for (size_t index = 0; index < output_elements; ++index) {
    if (std::memcmp(&raw_output[index], &decoded_output[index],
                    sizeof(float)) != 0 &&
        first_mismatch == output_elements) {
      first_mismatch = index;
    }
    const double absolute = std::abs(static_cast<double>(raw_output[index]) -
                                     decoded_output[index]);
    const double denominator =
        std::max(std::abs(static_cast<double>(raw_output[index])), 1.0e-12);
    *max_abs = std::max(*max_abs, absolute);
    *max_rel = std::max(*max_rel, absolute / denominator);
  }
  if (first_mismatch != output_elements) {
    std::cerr << "FAIL raw-vs-decoded GEMM not bit-exact: first_index="
              << first_mismatch << " raw=" << raw_output[first_mismatch]
              << " decoded=" << decoded_output[first_mismatch]
              << " max_abs=" << *max_abs << " max_rel=" << *max_rel
              << "\n";
    return false;
  }
  std::cout << "PASS GEMM raw B vs decoded B bit-exact; max_abs=" << *max_abs
            << " max_rel=" << *max_rel << "\n";

  std::vector<std::pair<int64_t, int64_t>> samples = {
      {0, 0}, {0, options.n / 2}, {0, options.n - 1},
      {options.m / 2, 0}, {options.m / 2, options.n / 2},
      {options.m / 2, options.n - 1}, {options.m - 1, 0},
      {options.m - 1, options.n / 2}, {options.m - 1, options.n - 1}};
  const uint16_t* weight =
      reinterpret_cast<const uint16_t*>(weight_bytes.data());
  *cpu_max_rel = 0.0;
  double cpu_max_abs = 0.0;
  double cpu_max_scaled = 0.0;
  for (const auto& sample : samples) {
    float reference = 0.0f;
    for (int64_t inner = 0; inner < options.k; ++inner) {
      reference +=
          bf16ToFloat(activation[sample.first * options.k + inner]) *
          bf16ToFloat(weight[inner * options.n + sample.second]);
    }
    const float actual = raw_output[sample.first * options.n + sample.second];
    const double absolute =
        std::abs(static_cast<double>(actual) - reference);
    const double scale =
        std::max(std::abs(static_cast<double>(reference)),
                 1.0e-2 * std::sqrt(static_cast<double>(options.k)));
    cpu_max_abs = std::max(cpu_max_abs, absolute);
    *cpu_max_rel = std::max(
        *cpu_max_rel,
        absolute /
            std::max(std::abs(static_cast<double>(reference)), 1.0e-12));
    cpu_max_scaled = std::max(cpu_max_scaled, absolute / scale);
  }
  std::cout << "CPU fp32 sampled reference: samples=" << samples.size()
            << " max_abs=" << cpu_max_abs
            << " max_relative=" << *cpu_max_rel
            << " max_scaled_relative=" << cpu_max_scaled
            << " scaled_tolerance=" << options.cpu_rel_tolerance << "\n";
  if (!std::isfinite(*cpu_max_rel) || !std::isfinite(cpu_max_scaled) ||
      cpu_max_scaled > options.cpu_rel_tolerance) {
    std::cerr << "FAIL CPU fp32 sampled reference tolerance\n";
    return false;
  }
  std::cout << "PASS CPU fp32 sampled reference\n";
  return true;
}

bool maybeFlush(bool cold, uint32_t* flush_device, hipStream_t stream,
                uint32_t salt) {
  if (!cold) {
    return true;
  }
  const size_t words = kFlushBytes / sizeof(uint32_t);
  hipLaunchKernelGGL(flushKernel, dim3(1024), dim3(256), 0, stream,
                     flush_device, words, salt);
  return checkHip(hipGetLastError(), "flushKernel launch");
}

template <typename Enqueue>
bool benchmarkOperation(const std::string& name, bool cold, int warmup,
                        int measure, uint32_t* flush_device,
                        hipStream_t stream, Enqueue enqueue, Stats* stats) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    if (!maybeFlush(cold, flush_device, stream,
                    static_cast<uint32_t>(iteration)) ||
        !enqueue() ||
        !checkHip(hipStreamSynchronize(stream), "warmup synchronize")) {
      std::cerr << "ERROR: " << name << " warmup iteration " << iteration
                << " failed\n";
      return false;
    }
  }
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (!checkHip(hipEventCreate(&start), "hipEventCreate(start)") ||
      !checkHip(hipEventCreate(&stop), "hipEventCreate(stop)")) {
    return false;
  }
  std::vector<double> times;
  times.reserve(measure);
  bool ok = true;
  for (int iteration = 0; iteration < measure; ++iteration) {
    if (!maybeFlush(cold, flush_device, stream,
                    static_cast<uint32_t>(warmup + iteration)) ||
        !checkHip(hipEventRecord(start, stream), "hipEventRecord(start)") ||
        !enqueue() ||
        !checkHip(hipEventRecord(stop, stream), "hipEventRecord(stop)") ||
        !checkHip(hipEventSynchronize(stop), "hipEventSynchronize(stop)")) {
      std::cerr << "ERROR: " << name << " measurement iteration "
                << iteration << " failed\n";
      ok = false;
      break;
    }
    float elapsed = 0.0f;
    if (!checkHip(hipEventElapsedTime(&elapsed, start, stop),
                  "hipEventElapsedTime")) {
      ok = false;
      break;
    }
    times.push_back(elapsed);
  }
  if (!checkHip(hipEventDestroy(start), "hipEventDestroy(start)")) {
    ok = false;
  }
  if (!checkHip(hipEventDestroy(stop), "hipEventDestroy(stop)")) {
    ok = false;
  }
  if (!ok || static_cast<int>(times.size()) != measure) {
    return false;
  }
  *stats = calculateStats(std::move(times));
  return true;
}

bool runTimingSet(bool cold, const Options& options,
                  hipans::StackDeviceMemory& stack,
                  const hipans::ANSCodecConfig& config, VendorGemm& gemm,
                  const uint16_t* activation_device,
                  const void* raw_weight_device,
                  const void* compressed_device, void* decoded_device,
                  float* output_device, uint8_t* success_device,
                  uint32_t* size_device, uint32_t* flush_device,
                  uint32_t raw_bytes, hipStream_t stream,
                  TimingResult* timing) {
  const std::string mode = cold ? "cold" : "hot";
  std::cout << "----- timing cache=" << mode << " -----\n";
  if (!benchmarkOperation(
          "t_gemm", cold, options.warmup, options.measure, flush_device,
          stream,
          [&]() {
            return gemm.run(activation_device, raw_weight_device,
                            output_device, stream);
          },
          &timing->gemm) ||
      !benchmarkOperation(
          "t_decode", cold, options.warmup, options.measure, flush_device,
          stream,
          [&]() {
            return enqueueDecode(stack, config, compressed_device,
                                 decoded_device, raw_bytes, success_device,
                                 size_device, stream);
          },
          &timing->decode) ||
      !benchmarkOperation(
          "t_decode_gemm", cold, options.warmup, options.measure,
          flush_device, stream,
          [&]() {
            return enqueueDecode(stack, config, compressed_device,
                                 decoded_device, raw_bytes, success_device,
                                 size_device, stream) &&
                   gemm.run(activation_device, decoded_device, output_device,
                            stream);
          },
          &timing->decode_gemm)) {
    return false;
  }
  timing->ratio = timing->decode_gemm.median / timing->gemm.median;
  std::cout << std::fixed << std::setprecision(6)
            << mode << " t_gemm_ms median/p5/p95=" << timing->gemm.median
            << "/" << timing->gemm.p5 << "/" << timing->gemm.p95 << "\n"
            << mode << " t_decode_ms median/p5/p95=" << timing->decode.median
            << "/" << timing->decode.p5 << "/" << timing->decode.p95 << "\n"
            << mode << " t_dec_gemm_ms median/p5/p95="
            << timing->decode_gemm.median << "/" << timing->decode_gemm.p5
            << "/" << timing->decode_gemm.p95 << "\n"
            << mode << " ratio=median(t_dec_gemm)/median(t_gemm)="
            << timing->ratio << "\n";
  return true;
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '\"') {
      escaped += "\"\"";
    } else {
      escaped += character;
    }
  }
  escaped += '\"';
  return escaped;
}

std::string timestampNow() {
  const std::time_t now = std::time(nullptr);
  std::tm local {};
  localtime_r(&now, &local);
  char buffer[64];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", &local);
  return buffer;
}

bool appendCsv(const Options& options, const std::string& hostname,
               const std::string& gpu_arch, const std::string& rocm_version,
               const std::string& hipans_commit,
               const std::string& benchmark_commit, const VendorGemm& gemm,
               uint64_t source_bytes, const std::string& fill_method,
               uint64_t raw_bytes, uint64_t compressed_bytes,
               double compression_ratio, const std::string& correctness,
               const std::string& run_status, const std::string& cache,
               const TimingResult& timing, double gemm_max_abs,
               double gemm_max_rel, double cpu_max_rel,
               const GpuSample& begin, const GpuSample& middle,
               const GpuSample& end) {
  bool needs_header = true;
  struct stat info {};
  if (stat(options.csv.c_str(), &info) == 0 && info.st_size > 0) {
    needs_header = false;
  }
  std::ofstream output(options.csv, std::ios::app);
  if (!output) {
    std::cerr << "ERROR: cannot append CSV: " << options.csv << "\n";
    return false;
  }
  if (needs_header) {
    output << "timestamp,hostname,gpu_index,gpu_arch,rocm_version,"
              "hipans_commit,benchmark_commit,build_flags,vendor_library,"
              "vendor_version,algorithm,workspace_bytes,weight_file,"
              "weight_source_bytes,weight_fill_method,activation_seed,m,n,k,"
              "b_raw_bytes,b_compressed_bytes,compression_ratio_raw_over_compressed,"
              "correctness,run_status,warmup_iters,measure_iters,cache,"
              "t_gemm_ms_median,t_gemm_ms_p5,t_gemm_ms_p95,"
              "t_decode_ms_median,t_decode_ms_p5,t_decode_ms_p95,"
              "t_dec_gemm_ms_median,t_dec_gemm_ms_p5,t_dec_gemm_ms_p95,ratio,"
              "gemm_raw_decoded_max_abs,gemm_raw_decoded_max_rel,"
              "cpu_sample_max_rel,sclk_begin_mhz,sclk_mid_mhz,sclk_end_mhz,"
              "mclk_begin_mhz,mclk_mid_mhz,mclk_end_mhz,temp_begin_c,"
              "temp_mid_c,temp_end_c\n";
  }
  output << std::setprecision(10) << csvEscape(timestampNow()) << ','
         << csvEscape(hostname) << ',' << options.gpu_index << ','
         << csvEscape(gpu_arch) << ',' << csvEscape(rocm_version) << ','
         << csvEscape(hipans_commit) << ',' << csvEscape(benchmark_commit) << ','
         << csvEscape(kBuildFlags) << ',' << csvEscape(gemm.backend()) << ','
         << csvEscape(gemm.version()) << ',' << csvEscape(gemm.algorithm())
         << ',' << gemm.workspaceBytes() << ','
         << csvEscape(options.weight_file) << ',' << source_bytes << ','
         << csvEscape(fill_method) << ',' << options.seed << ',' << options.m
         << ',' << options.n << ',' << options.k << ',' << raw_bytes << ','
         << compressed_bytes << ',' << compression_ratio << ','
         << correctness << ',' << run_status << ',' << options.warmup << ','
         << options.measure << ',' << cache << ',' << timing.gemm.median << ','
         << timing.gemm.p5 << ',' << timing.gemm.p95 << ','
         << timing.decode.median << ',' << timing.decode.p5 << ','
         << timing.decode.p95 << ',' << timing.decode_gemm.median << ','
         << timing.decode_gemm.p5 << ',' << timing.decode_gemm.p95 << ','
         << timing.ratio << ',' << gemm_max_abs << ',' << gemm_max_rel << ','
         << cpu_max_rel << ',' << optionalNumber(begin.sclk_mhz) << ','
         << optionalNumber(middle.sclk_mhz) << ','
         << optionalNumber(end.sclk_mhz) << ','
         << optionalNumber(begin.mclk_mhz) << ','
         << optionalNumber(middle.mclk_mhz) << ','
         << optionalNumber(end.mclk_mhz) << ','
         << optionalNumber(begin.temperature_c) << ','
         << optionalNumber(middle.temperature_c) << ','
         << optionalNumber(end.temperature_c) << '\n';
  return static_cast<bool>(output);
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parseOptions(argc, argv, &options)) {
    return 2;
  }
  ScopedStderrCapture stderr_capture;

  const char* slurm_job = std::getenv("SLURM_JOB_ID");
  const char* visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
  char hostname_buffer[256] = {};
  if (gethostname(hostname_buffer, sizeof(hostname_buffer) - 1) != 0) {
    std::strcpy(hostname_buffer, "unknown");
  }
  const std::string hostname = hostname_buffer;
  std::cout << "============================================================\n"
            << "hipANS decode + vendor bf16 GEMM separated baseline\n"
            << "hostname: " << hostname << "\n"
            << "SLURM_JOB_ID: " << (slurm_job ? slurm_job : "unset") << "\n"
            << "HIP_VISIBLE_DEVICES: "
            << (visible_devices ? visible_devices : "unset") << "\n"
            << "physical_gpu_index: " << options.gpu_index << "\n"
            << "shape M,N,K: " << options.m << ',' << options.n << ','
            << options.k << "\n"
            << "layout: row-major A[M,K], B[K,N], C[M,N]\n"
            << "types: A=bf16 B=bf16 accumulation=fp32 output=fp32\n"
            << "ANS config: probBits=10 useChecksum=false numInBatch=1\n"
            << "weight_file: " << options.weight_file << "\n"
            << "weight_fill_rule: prefix truncate if sufficient; otherwise "
               "cyclic repetition\n"
            << "activation_distribution: normal(mean=0,stddev=0.5) rounded "
               "to bf16\n"
            << "activation_seed: " << options.seed << "\n"
            << "warmup_iters_per_metric_per_cache_mode: " << options.warmup
            << "\n"
            << "measure_iters_per_metric_per_cache_mode: " << options.measure
            << "\n"
            << "cache_modes: cold,hot\n"
            << "cold_flush_bytes_outside_timing: " << kFlushBytes << "\n"
            << "StackDeviceMemory_bytes: " << kStackBytes << "\n"
            << "hipBLASLt_max_workspace_bytes: " << kLtWorkspaceBytes << "\n"
            << "cpu_sample_relative_tolerance: "
            << options.cpu_rel_tolerance << "\n"
            << "ratio_definition: median(t_dec_gemm)/median(t_gemm)\n"
            << "compression_ratio_definition: raw_B_bytes/compressed_B_bytes\n"
            << "build_flags: " << kBuildFlags << "\n"
            << "CSV: " << options.csv << "\n"
            << "============================================================\n";

  if (slurm_job == nullptr || hostname.rfind("hacchead", 0) == 0) {
    std::cerr << "ERROR: refusing to run outside a Slurm compute allocation\n";
    return 3;
  }
  if (visible_devices == nullptr ||
      std::string(visible_devices) != std::to_string(options.gpu_index)) {
    std::cerr << "ERROR: HIP_VISIBLE_DEVICES must exactly match --gpu-index\n";
    return 3;
  }
  int device_count = 0;
  if (!checkHip(hipGetDeviceCount(&device_count), "hipGetDeviceCount") ||
      device_count != 1 || !checkHip(hipSetDevice(0), "hipSetDevice(0)")) {
    std::cerr << "ERROR: expected exactly one HIP-visible GPU\n";
    return 3;
  }
  hipDeviceProp_t properties {};
  if (!checkHip(hipGetDeviceProperties(&properties, 0),
                "hipGetDeviceProperties")) {
    return 3;
  }
  const std::string gpu_arch = properties.gcnArchName;
  const std::string rocm_version = commandValue(
      "hipcc --version | sed -n 's/.*HIP version: //p' | head -1", "unknown");
  const std::string hipans_commit = commandValue(
      std::string("git -C ") + kRepo + " rev-parse HEAD", "unknown");
  const std::string benchmark_commit = commandValue(
      std::string("git -C ") + kBenchDir + " rev-parse HEAD", "unknown");
  std::cout << "GPU name: " << properties.name << "\n"
            << "GPU arch: " << gpu_arch << "\n"
            << "ROCm/HIP version: " << rocm_version << "\n"
            << "hipANS commit: " << hipans_commit << "\n"
            << "benchmark commit: " << benchmark_commit << "\n";

  size_t activation_elements = 0;
  size_t weight_elements = 0;
  size_t output_elements = 0;
  if (!checkedElements(options.m, options.k, &activation_elements) ||
      !checkedElements(options.k, options.n, &weight_elements) ||
      !checkedElements(options.m, options.n, &output_elements) ||
      weight_elements > std::numeric_limits<uint32_t>::max() / 2ULL) {
    std::cerr << "ERROR: shape is too large or invalid\n";
    return 4;
  }
  const uint32_t weight_bytes = static_cast<uint32_t>(weight_elements * 2ULL);
  std::vector<uint8_t> weight_host;
  uint64_t source_bytes = 0;
  std::string fill_method;
  if (!readWeight(options, weight_bytes, &weight_host, &source_bytes,
                  &fill_method)) {
    return 4;
  }
  const std::vector<uint16_t> activation_host =
      makeActivation(activation_elements, options.seed);
  std::cout << "weight_source_bytes: " << source_bytes << "\n"
            << "weight_requested_bytes: " << weight_bytes << "\n"
            << "weight_fill_method: " << fill_method << "\n";

  DeviceBuffer<uint16_t> activation_device;
  DeviceBuffer<uint8_t> weight_device;
  DeviceBuffer<uint8_t> compressed_device;
  DeviceBuffer<uint8_t> decoded_device;
  DeviceBuffer<uint32_t> compressed_size_device;
  DeviceBuffer<uint8_t> decode_success_device;
  DeviceBuffer<uint32_t> decode_size_device;
  DeviceBuffer<float> raw_output_device;
  DeviceBuffer<float> decoded_output_device;
  DeviceBuffer<uint32_t> flush_device;
  const uint32_t compressed_capacity = hipans::getMaxCompressedSize(weight_bytes);
  if (!activation_device.allocate(activation_elements, "activation A") ||
      !weight_device.allocate(weight_bytes, "raw B bytes") ||
      !compressed_device.allocate(compressed_capacity, "compressed B") ||
      !decoded_device.allocate(weight_bytes, "decoded B bytes") ||
      !compressed_size_device.allocate(1, "compressed size") ||
      !decode_success_device.allocate(1, "decode success") ||
      !decode_size_device.allocate(1, "decode size") ||
      !raw_output_device.allocate(output_elements, "raw GEMM output") ||
      !decoded_output_device.allocate(output_elements,
                                      "decoded GEMM output") ||
      !flush_device.allocate(kFlushBytes / sizeof(uint32_t),
                             "L2 flush buffer")) {
    return 5;
  }

  hipStream_t stream = nullptr;
  if (!checkHip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
                "hipStreamCreateWithFlags") ||
      !checkHip(hipMemcpyAsync(activation_device.data(), activation_host.data(),
                               activation_device.bytes(),
                               hipMemcpyHostToDevice, stream),
                "activation H2D") ||
      !checkHip(hipMemcpyAsync(weight_device.data(), weight_host.data(),
                               weight_host.size(), hipMemcpyHostToDevice,
                               stream),
                "weight H2D") ||
      !checkHip(hipStreamSynchronize(stream), "initial H2D synchronize")) {
    return 5;
  }

  int return_code = 0;
  {
    auto stack = hipans::makeStackMemory(kStackBytes);
    const hipans::ANSCodecConfig ans_config(10, false);
    uint32_t compressed_bytes = 0;
    if (!encodeWeight(stack, ans_config, weight_device.data(), weight_bytes,
                      compressed_device.data(), compressed_size_device.data(),
                      stream, &compressed_bytes) ||
        compressed_bytes == 0 || compressed_bytes > compressed_capacity) {
      std::cerr << "ERROR: invalid compressed size " << compressed_bytes
                << " capacity=" << compressed_capacity << "\n";
      return_code = 6;
    }
    const double compression_ratio =
        static_cast<double>(weight_bytes) / compressed_bytes;
    std::cout << "B_raw_bytes: " << weight_bytes << "\n"
              << "B_compressed_bytes: " << compressed_bytes << "\n"
              << "compression_ratio_raw_over_compressed: "
              << std::setprecision(9) << compression_ratio << "\n";

    VendorGemm gemm;
    if (return_code == 0 && !gemm.initialize(options.m, options.n, options.k)) {
      return_code = 7;
    }
    if (return_code == 0) {
      std::cout << "vendor_library: " << gemm.backend() << "\n"
                << "vendor_version: " << gemm.version() << "\n"
                << "selected_algorithm: " << gemm.algorithm() << "\n"
                << "selected_workspace_bytes: " << gemm.workspaceBytes()
                << "\n";
    }

    bool correctness_pass = false;
    double gemm_max_abs = 0.0;
    double gemm_max_rel = 0.0;
    double cpu_max_rel = 0.0;
    if (return_code == 0) {
      correctness_pass = verifyDecode(
          stack, ans_config, weight_host, compressed_device.data(),
          decoded_device.data(), decode_success_device.data(),
          decode_size_device.data(), stream);
      if (correctness_pass) {
        correctness_pass = verifyGemm(
            gemm, options, activation_host, weight_host,
            activation_device.data(), weight_device.data(),
            decoded_device.data(), raw_output_device.data(),
            decoded_output_device.data(), stream, &gemm_max_abs,
            &gemm_max_rel, &cpu_max_rel);
      }
      if (!correctness_pass) {
        return_code = 8;
      }
    }

    const CommandResult perf = runCommand(
        "rocm-smi -d " + std::to_string(options.gpu_index) +
        " --setperflevel high");
    std::cout << "----- performance level request -----\n"
              << perf.output;
    if (perf.exit_code == 0) {
      std::cout << "frequency_control: requested performance level high\n";
    } else {
      std::cout << "frequency_control: 未锁频\n";
    }

    GpuSample precheck;
    GpuSample begin;
    GpuSample middle;
    GpuSample end;
    bool contended = false;
    TimingResult cold;
    TimingResult hot;
    if (return_code == 0) {
      precheck = sampleGpu("pre-measurement", options.gpu_index);
      if (!precheck.ok || !precheck.utilization ||
          *precheck.utilization != 0.0) {
        contended = true;
        std::cout << "WARNING: pre-measurement GPU utilization is not "
                     "confirmed as 0%; run marked CONTENDED\n";
      } else {
        std::cout << "pre-measurement GPU utilization: 0%\n";
      }
      begin = sampleGpu("measurement begin", options.gpu_index);
      if (!runTimingSet(true, options, stack, ans_config, gemm,
                        activation_device.data(), weight_device.data(),
                        compressed_device.data(), decoded_device.data(),
                        raw_output_device.data(), decode_success_device.data(),
                        decode_size_device.data(), flush_device.data(),
                        weight_bytes, stream, &cold)) {
        return_code = 9;
      }
      middle = sampleGpu("measurement midpoint", options.gpu_index);
      if (return_code == 0 &&
          !runTimingSet(false, options, stack, ans_config, gemm,
                        activation_device.data(), weight_device.data(),
                        compressed_device.data(), decoded_device.data(),
                        raw_output_device.data(), decode_success_device.data(),
                        decode_size_device.data(), flush_device.data(),
                        weight_bytes, stream, &hot)) {
        return_code = 9;
      }
      end = sampleGpu("measurement end", options.gpu_index);
    }

    const bool overflow =
        stderr_capture.captured().find(kOverflowWarning) != std::string::npos;
    if (overflow) {
      std::cerr << "ERROR: StackDeviceMemory overflow warning detected; "
                   "run marked INVALID\n";
      return_code = 10;
    }
    const std::string run_status =
        return_code != 0 ? "INVALID" : (contended ? "CONTENDED" : "OK");
    const std::string correctness = correctness_pass ? "PASS" : "FAIL";
    std::cout << "correctness: " << correctness << "\n"
              << "run_status: " << run_status << "\n";
    if (return_code == 0) {
      if (!appendCsv(options, hostname, gpu_arch, rocm_version, hipans_commit,
                     benchmark_commit, gemm, source_bytes, fill_method,
                     weight_bytes, compressed_bytes, compression_ratio,
                     correctness, run_status, "cold", cold, gemm_max_abs,
                     gemm_max_rel, cpu_max_rel, begin, middle, end) ||
          !appendCsv(options, hostname, gpu_arch, rocm_version, hipans_commit,
                     benchmark_commit, gemm, source_bytes, fill_method,
                     weight_bytes, compressed_bytes, compression_ratio,
                     correctness, run_status, "hot", hot, gemm_max_abs,
                     gemm_max_rel, cpu_max_rel, begin, middle, end)) {
        return_code = 11;
      } else {
        std::cout << "CSV rows appended: " << options.csv << "\n";
      }
    }
  }
  if (stream != nullptr &&
      !checkHip(hipStreamDestroy(stream), "hipStreamDestroy") &&
      return_code == 0) {
    return_code = 12;
  }
  std::cout << "FINAL RESULT: " << (return_code == 0 ? "PASS" : "FAIL")
            << " rc=" << return_code << "\n";
  return return_code;
}
