// CUDA C++ counterpart to the Practical 2 cuda-oxide port.
// See README.md for build instructions and execution flow.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef CPP_CPU_TEST
#include <cuda_runtime.h>
#include <curand.h>
#define HD __host__ __device__
#else
#define HD
#endif

constexpr unsigned BLOCK_SIZE = 128;
constexpr std::uint64_t SEED = 1234;

struct Words4 { std::uint32_t x, y, z, w; };
struct NormalPair { float x, y; };

/// Run Philox4x32-10 with unsigned wraparound in the key schedule.
HD Words4 philox(Words4 c, std::uint32_t k0, std::uint32_t k1) {
    for (int round = 0; round < 10; ++round) {
        const std::uint64_t a = std::uint64_t{0xD2511F53u} * c.x;
        const std::uint64_t b = std::uint64_t{0xCD9E8D57u} * c.z;
        c = {static_cast<std::uint32_t>(b >> 32) ^ c.y ^ k0,
             static_cast<std::uint32_t>(b),
             static_cast<std::uint32_t>(a >> 32) ^ c.w ^ k1,
             static_cast<std::uint32_t>(a)};
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

/// Convert a Philox counter to a Box–Muller pair using midpoint uniforms in (0, 1).
HD NormalPair normal_pair(std::uint32_t path, std::uint32_t step,
                          std::uint32_t batch, std::uint64_t seed) {
    const Words4 c = philox({step, path, batch, 0},
                           static_cast<std::uint32_t>(seed),
                           static_cast<std::uint32_t>(seed >> 32));
    const float u1 = (static_cast<float>(c.x >> 9) + 0.5f) * (1.0f / 8388608.0f);
    const float u2 = (static_cast<float>(c.y >> 9) + 0.5f) * (1.0f / 8388608.0f);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float angle = 6.283185482025146484375f * u2; // Rust f32::consts::TAU
    return {radius * cosf(angle), radius * sinf(angle)};
}

/// Compute the sample mean and its standard error with double accumulators.
std::pair<double, double> statistics(const std::vector<float>& values) {
    double sum = 0.0, square_sum = 0.0;
    for (float x : values) {
        const double d = x;
        sum += d;
        square_sum += d * d;
    }
    const double n = static_cast<double>(values.size());
    const double mean = sum / n;
    return {mean, std::sqrt(std::max(0.0, square_sum / n - mean * mean) / n)};
}

/// Simulate one payoff on the CPU with the selected buffered layout.
float reference(const std::vector<float>& z, std::size_t i, std::size_t steps,
                std::size_t width, unsigned version, const float* p) {
    const std::size_t block = i / width, lane = i % width;
    const std::size_t stride = version == 1 ? width : 1;
    std::size_t ind = block * 2 * steps * width
                    + (version == 1 ? lane : lane * 2 * steps);
    float s1 = 1.0f, s2 = 1.0f;
    for (std::size_t n = 0; n < steps; ++n) {
        const float y1 = z[ind]; ind += stride;
        const float y2 = p[0] * y1 + p[1] * z[ind]; ind += stride;
        s1 *= p[2] + p[3] * y1;
        s2 *= p[2] + p[3] * y2;
    }
    return fabsf(s1 - 1.0f) < 0.1f && fabsf(s2 - 1.0f) < 0.1f ? p[4] : 0.0f;
}

#ifndef CPP_CPU_TEST

// Rust #[constant] declarations, initialized by the host before launching.
__constant__ std::uint32_t N;
__constant__ float PARAMS[5]; // rho, alpha, con1, con2, discount
__constant__ float ABC[3];

/// Simulate one payoff per active GPU thread using one of the two buffered layouts.
__global__ void pathcalc(const float* z, float* out, std::size_t paths,
                         unsigned version) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= paths) return;
    const std::size_t nsteps = N;
    const std::size_t width = blockDim.x, lane = threadIdx.x, block = blockIdx.x;
    const std::size_t stride = version == 1 ? width : 1;
    std::size_t ind = block * (2 * nsteps * width)
                    + (version == 1 ? lane : lane * 2 * nsteps);
    float s1 = 1.0f, s2 = 1.0f;
    for (std::size_t n = 0; n < nsteps; ++n) {
        const float y1 = z[ind]; ind += stride;
        const float y2 = PARAMS[0] * y1 + PARAMS[1] * z[ind]; ind += stride;
        s1 *= PARAMS[2] + PARAMS[3] * y1;
        s2 *= PARAMS[2] + PARAMS[3] * y2;
    }
    out[i] = fabsf(s1 - 1.0f) < 0.1f && fabsf(s2 - 1.0f) < 0.1f
           ? PARAMS[4] : 0.0f;
}

/// Average 200 polynomial evaluations per active GPU thread.
__global__ void polynomial(const float* z, float* out, std::size_t paths) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= paths) return;
    const std::size_t width = blockDim.x;
    const std::size_t base = std::size_t(blockIdx.x) * 200 * width + threadIdx.x;
    float sum = 0.0f;
    for (std::size_t j = 0; j < 200; ++j) {
        const float x = z[base + j * width];
        sum += ABC[0] * x * x + ABC[1] * x + ABC[2];
    }
    out[i] = sum / 200.0f;
}

/// Simulate payoffs with a grid-stride loop and generate normals in registers.
__global__ void pathcalc_device(float* out, std::size_t paths,
                                std::uint64_t seed, std::uint32_t batch) {
    std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
    while (i < paths) {
        float s1 = 1.0f, s2 = 1.0f;
        for (std::uint32_t step = 0; step < N; ++step) {
            const NormalPair z = normal_pair(static_cast<std::uint32_t>(i), step, batch, seed);
            const float y2 = PARAMS[0] * z.x + PARAMS[1] * z.y;
            s1 *= PARAMS[2] + PARAMS[3] * z.x;
            s2 *= PARAMS[2] + PARAMS[3] * y2;
        }
        out[i] = fabsf(s1 - 1.0f) < 0.1f && fabsf(s2 - 1.0f) < 0.1f
               ? PARAMS[4] : 0.0f;
        i += stride;
    }
}

/// Convert a failed CUDA Runtime call to an exception with its source line.
void check_cuda(cudaError_t code, const char* expression, int line) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(expression) + " at line "
            + std::to_string(line) + ": " + cudaGetErrorString(code));
}
/// Convert a failed cuRAND call to an exception with its source line.
void check_curand(curandStatus_t code, const char* expression, int line) {
    if (code != CURAND_STATUS_SUCCESS)
        throw std::runtime_error(std::string(expression) + " at line "
            + std::to_string(line) + ": cuRAND status " + std::to_string(int(code)));
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __LINE__)
#define CURAND_CHECK(expr) check_curand((expr), #expr, __LINE__)

// RAII wrappers mirror DeviceBuffer, timed events and RNG ownership in Rust.
struct DeviceBuffer {
    float* data = nullptr;
    std::size_t len;
    /// Allocate a zero-initialized array of floats on the device.
    explicit DeviceBuffer(std::size_t n) : len(n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(float))
            throw std::overflow_error("allocation byte count overflow");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float)));
        const cudaError_t status = cudaMemset(data, 0, n * sizeof(float));
        if (status != cudaSuccess) { cudaFree(data); data = nullptr; }
        CUDA_CHECK(status);
    }
    /// Free the owned device allocation.
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    /// Copy the device array to host memory and wait for the copy.
    std::vector<float> to_host(cudaStream_t stream) const {
        std::vector<float> result(len);
        CUDA_CHECK(cudaMemcpyAsync(result.data(), data, len * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return result;
    }
};
struct Event {
    cudaEvent_t value = nullptr;
    /// Create a CUDA event with timing enabled.
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    /// Destroy the owned CUDA event.
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct Generator {
    curandGenerator_t value = nullptr;
    /// Create a seeded cuRAND generator on the given stream.
    explicit Generator(cudaStream_t stream) {
        CURAND_CHECK(curandCreateGenerator(&value, CURAND_RNG_PSEUDO_DEFAULT));
        try {
            CURAND_CHECK(curandSetStream(value, stream));
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(value, SEED));
        } catch (...) { curandDestroyGenerator(value); value = nullptr; throw; }
    }
    /// Destroy the owned cuRAND generator.
    ~Generator() { if (value) curandDestroyGenerator(value); }
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
};

/// Time one queued operation with CUDA events, excluding any requested warm-up.
template <typename F>
float elapsed_ms(F fn, cudaStream_t stream, bool warmup) {
    if (warmup) { fn(); CUDA_CHECK(cudaStreamSynchronize(stream)); }
    Event start, stop;
    CUDA_CHECK(cudaEventRecord(start.value, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop.value, stream));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.value, stop.value));
    return ms;
}

/// Read a positive decimal count that fits the CUDA kernel's 32-bit parameters.
std::uint32_t positive_count(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected a positive decimal integer: " + text);
    const auto value = std::stoull(text);
    if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("paths and steps must fit nonzero u32 values");
    return static_cast<std::uint32_t>(value);
}
/// Multiply two sizes and reject overflow before allocating.
std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("sample count overflow");
    return a * b;
}

struct RunConfig {
    std::string mode;
    bool verify;
    std::size_t paths;
    std::uint32_t steps;
    std::size_t blocks;
    std::size_t normal_count;
    std::size_t normal_bytes;
};

/// Print command-line syntax and defaults.
void print_help() {
    std::cout << "Usage: monte-carlo --help\n"
              << "       monte-carlo [buffered [paths [steps [verify]]]]\n"
              << "       monte-carlo device [paths [steps]]\n"
              << "       monte-carlo polynomial [paths [steps]] (always uses 200 samples per path)\n"
              << "Defaults: buffered, 131072 paths, 100 steps, block size 128, seed 1234\n"
              << "verify: compare buffered payoffs with the CPU reference\n";
}

/// Compute the launch and buffer sizes from the positional arguments.
RunConfig make_config(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "buffered";
    if (mode != "buffered" && mode != "device" && mode != "polynomial")
        throw std::invalid_argument("mode must be buffered, device, or polynomial");
    if (argc > (mode == "buffered" ? 5 : 4))
        throw std::invalid_argument("too many arguments; use --help");
    const bool verify = argc == 5;
    if (verify && std::string_view(argv[4]) != "verify")
        throw std::invalid_argument("fourth argument must be verify; use --help");
    const std::size_t paths = argc > 2 ? positive_count(argv[2]) : 131072;
    const std::uint32_t steps = argc > 3 ? positive_count(argv[3]) : 100;
    const std::size_t blocks = (paths - 1) / BLOCK_SIZE + 1;
    const std::size_t padded = checked_mul(blocks, BLOCK_SIZE);
    const std::size_t samples_per_path = mode == "polynomial" ? 200 : checked_mul(steps, 2);
    const std::size_t count = checked_mul(padded, samples_per_path);
    return {mode, verify, paths, steps, blocks, count, checked_mul(count, sizeof(float))};
}

/// Upload the step count, asset parameters and polynomial coefficients.
void upload_constants(std::uint32_t steps, float (&p)[5]) {
    const float rho = 0.5f, dt = 1.0f / static_cast<float>(steps);
    p[0] = rho;
    p[1] = sqrtf(1.0f - rho * rho);
    p[2] = 1.0f + 0.05f * dt;
    p[3] = 0.1f * sqrtf(dt);
    p[4] = expf(-0.05f);
    const float abc[3] = {2.0f, 3.0f, 4.0f};
    CUDA_CHECK(cudaMemcpyToSymbol(N, &steps, sizeof(steps)));
    CUDA_CHECK(cudaMemcpyToSymbol(PARAMS, p, sizeof(p)));
    CUDA_CHECK(cudaMemcpyToSymbol(ABC, abc, sizeof(abc)));
}

/// Select and report the occupancy-based grid for the Philox kernel.
unsigned device_grid() {
    int resident = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, pathcalc_device, BLOCK_SIZE, 0));
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    const auto grid = checked_mul(static_cast<std::size_t>(resident),
                                  static_cast<std::size_t>(props.multiProcessorCount));
    if (resident <= 0 || grid == 0 || grid > static_cast<std::size_t>(props.maxGridSize[0]))
        throw std::runtime_error("invalid occupancy launch grid");
    std::cout << "SMs: " << props.multiProcessorCount << "; active blocks/SM: "
              << resident << "; grid blocks: " << grid << '\n';
    return static_cast<unsigned>(grid);
}

/// Time the Philox kernel and report payoff statistics and sample throughput.
void run_device(const RunConfig& cfg, DeviceBuffer& out, cudaStream_t stream) {
    const unsigned grid = device_grid();
    const float ms = elapsed_ms([&] {
        pathcalc_device<<<grid, BLOCK_SIZE, 0, stream>>>(out.data, cfg.paths, SEED, 0);
        CUDA_CHECK(cudaGetLastError());
    }, stream, true);
    const auto stats = statistics(out.to_host(stream));
    std::cout << "device: " << ms << " ms; mean " << stats.first
              << "; standard error " << stats.second << '\n';
    std::cout << "normal samples/s: " << std::scientific
              << 2.0 * cfg.steps * static_cast<double>(cfg.paths) / (ms * 1e-3) << '\n';
}

/// Fill the normal buffer using cuRAND and time its first generation call.
void generate_normals(DeviceBuffer& z, Generator& rng, cudaStream_t stream) {
    const float ms = elapsed_ms([&] {
        CURAND_CHECK(curandGenerateNormal(rng.value, z.data, z.len, 0.0f, 1.0f));
    }, stream, false);
    std::cout << "cuRAND: " << ms << " ms (includes first-call setup)\n";
}

/// Time the polynomial kernel and compare its statistics to Gaussian moments.
void run_polynomial(const RunConfig& cfg, const DeviceBuffer& z, DeviceBuffer& out,
                    cudaStream_t stream) {
    const float ms = elapsed_ms([&] {
        polynomial<<<static_cast<unsigned>(cfg.blocks), BLOCK_SIZE, 0, stream>>>(
            z.data, out.data, cfg.paths);
        CUDA_CHECK(cudaGetLastError());
    }, stream, true);
    const auto stats = statistics(out.to_host(stream));
    std::cout << "polynomial: " << ms << " ms; mean " << stats.first
              << "; standard error " << stats.second << "; expected 6\n";
    std::cout << "analytical standard error: " << std::sqrt(17.0 / (200.0 * cfg.paths)) << '\n';
}

/// Compare each GPU payoff to the scalar reference for the same memory layout.
void verify_payoffs(const std::vector<float>& values, const std::vector<float>& normals,
                    std::uint32_t steps, unsigned version, const float* p) {
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        const float expected = reference(normals, i, steps, BLOCK_SIZE, version, p);
        if (fabsf(values[i] - expected) > 1e-6f) ++mismatches;
    }
    if (mismatches) throw std::runtime_error(std::to_string(mismatches)
        + " CPU/GPU payoff mismatches; retry with --fmad=false to check FMA boundary effects");
    std::cout << "CPU reference: all " << values.size() << " payoffs matched\n";
}

/// Launch one buffered layout and optionally verify all its payoffs.
void run_layout(const RunConfig& cfg, const DeviceBuffer& z, DeviceBuffer& out,
                cudaStream_t stream, unsigned version, const std::vector<float>& host_z,
                const float* p) {
    const float ms = elapsed_ms([&] {
        pathcalc<<<static_cast<unsigned>(cfg.blocks), BLOCK_SIZE, 0, stream>>>(
            z.data, out.data, cfg.paths, version);
        CUDA_CHECK(cudaGetLastError());
    }, stream, true);
    const auto values = out.to_host(stream);
    const auto stats = statistics(values);
    const double traffic = (2.0 * cfg.steps + 1.0) * cfg.paths * sizeof(float);
    std::cout << "version " << version << ": " << ms << " ms; mean " << stats.first
              << "; standard error " << stats.second << "; effective "
              << traffic / (static_cast<double>(ms) * 1e6) << " GB/s\n";
    if (cfg.verify) verify_payoffs(values, host_z, cfg.steps, version, p);
}

/// Prepare the normal buffer and dispatch to the polynomial or payoff kernels.
void run_buffered(const RunConfig& cfg, DeviceBuffer& out, cudaStream_t stream, const float* p) {
    std::cout << "Normal buffer: " << cfg.normal_count << " floats, " << cfg.normal_bytes / 1e9
              << " GB (including padded lanes)\n";
    DeviceBuffer z(cfg.normal_count);
    Generator rng(stream);
    generate_normals(z, rng, stream);
    if (cfg.mode == "polynomial") {
        run_polynomial(cfg, z, out, stream);
        return;
    }
    const auto host_z = cfg.verify ? z.to_host(stream) : std::vector<float>{};
    for (unsigned version : {1u, 2u}) run_layout(cfg, z, out, stream, version, host_z, p);
}

/// Initialize the default CUDA stream and dispatch to the selected simulation.
void run(const RunConfig& cfg) {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream = nullptr;
    float params[5];
    upload_constants(cfg.steps, params);
    DeviceBuffer out(cfg.paths);
    std::cout << std::fixed << std::setprecision(8);
    if (cfg.mode == "device") run_device(cfg, out, stream);
    else run_buffered(cfg, out, stream, params);
}

/// Show help or run the simulation with the supplied positional arguments.
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            print_help();
            return 0;
        }
        run(make_config(argc, argv));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

#else // CPP_CPU_TEST

/// Throw when a CPU regression check fails.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

/// Check Philox against a known-answer vector.
void check_philox() {
    const Words4 c = philox({0, 0, 0, 0}, 0, 0);
    require(c.x == 0x6627e8d5u && c.y == 0xe169c58du
         && c.z == 0xbc57ac4cu && c.w == 0x9b00dbd8u, "Philox known-answer failed");
}

/// Verify that each layout visits every padded normal slot exactly once.
void check_layout_coverage() {
    for (std::size_t paths : {1u, 32u, 127u, 128u, 129u, 4097u}) {
        const std::size_t padded = ((paths + 127) / 128) * 128, steps = 3;
        for (unsigned version : {1u, 2u}) {
            std::vector<unsigned> seen(padded * 2 * steps, 0);
            for (std::size_t i = 0; i < padded; ++i) {
                const std::size_t stride = version == 1 ? 128 : 1;
                const std::size_t base = (i / 128) * 2 * steps * 128
                    + (version == 1 ? i % 128 : (i % 128) * 2 * steps);
                for (std::size_t j = 0; j < 2 * steps; ++j) ++seen.at(base + j * stride);
            }
            require(std::all_of(seen.begin(), seen.end(), [](unsigned x) { return x == 1; }),
                    "layout coverage failed");
        }
    }
}

/// Check the scalar payoff for both layouts with zero normal samples.
void check_zero_noise_payoff() {
    const float p[5] = {0.5f, sqrtf(0.75f), 1.0005f, 0.01f, expf(-0.05f)};
    const std::vector<float> zeros(256 * 2 * 100, 0.0f);
    for (unsigned version : {1u, 2u})
        for (std::size_t i = 0; i < 129; ++i)
            require(reference(zeros, i, 100, 128, version, p) == p[4], "zero-noise payoff failed");
}

/// Check the first, second and cross moments of Philox-generated normal pairs.
void check_normal_moments() {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    const unsigned count = 100000;
    for (unsigned i = 0; i < count; ++i) {
        const NormalPair z = normal_pair(i, 0, 0, SEED);
        const double x = z.x, y = z.y;
        sx += x; sy += y; sxx += x*x; syy += y*y; sxy += x*y;
    }
    require(std::abs(sx/count) < 0.02 && std::abs(sy/count) < 0.02, "normal means failed");
    require(std::abs(sxx/count-1) < 0.03 && std::abs(syy/count-1) < 0.03, "normal second moments failed");
    require(std::abs(sxy/count) < 0.02, "normal cross moment failed");
}

/// Check the reported mean and standard error for a two-point sample.
void check_statistics() {
    const auto stats = statistics({0.0f, 1.0f});
    require(stats.first == 0.5 && std::abs(stats.second - std::sqrt(0.125)) < 1e-15,
            "payoff statistics failed");
}

/// Run all CPU-only checks and report the first failure.
int main() {
    try {
        check_philox();
        check_layout_coverage();
        check_zero_noise_payoff();
        check_normal_moments();
        check_statistics();
        std::cout << "PASS: Philox known-answer, padded layout coverage, scalar payoff, normal moments, statistics\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
#endif
