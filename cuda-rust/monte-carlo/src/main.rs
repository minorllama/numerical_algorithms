//! Mike Giles's CUDA Practical 2, ported to Rust with cuda-oxide.
//! See README.md for the mapping, commands, and validation limitations.

use cuda_core::curand::RNG;
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};
use std::sync::Arc;

/// Kernels share host-populated constant memory, as in Giles's C++ code.
#[cuda_module]
mod kernels {
    use super::*;

    /// Number of Euler steps in each simulated path.
    #[constant]
    static N: ConstantMemory<u32> = ConstantMemory::UNINIT;
    /// [rho, alpha, 1+r*dt, sigma*sqrt(dt), exp(-r*T)].
    #[constant]
    static PARAMS: ConstantMemory<[f32; 5]> = ConstantMemory::UNINIT;
    /// Polynomial coefficients [a, b, c] for exercise 6.
    #[constant]
    static ABC: ConstantMemory<[f32; 3]> = ConstantMemory::UNINIT;

    /// Return the discounted payoff when both assets meet the terminal condition.
    fn payoff(s1: f32, s2: f32, discount: f32) -> f32 {
        if (s1 - 1.0).abs() < 0.1 && (s2 - 1.0).abs() < 0.1 {
            discount
        } else {
            0.0
        }
    }

    /// Price one path per active thread using the selected normal-buffer layout.
    #[kernel]
    pub fn pathcalc(z: &[f32], mut out: DisjointSlice<f32>, version: u32) {
        let Some(slot) = out.get_mut(thread::index_1d()) else {
            return;
        };
        let steps = N.get() as usize;
        let width = thread::blockDim_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let block_base = thread::blockIdx_x() as usize * (2 * steps * width);
        let (mut offset, stride) = if version == 1 {
            (block_base + lane, width)
        } else {
            (block_base + lane * 2 * steps, 1)
        };
        let p = PARAMS.get();
        let (mut s1, mut s2) = (1.0_f32, 1.0_f32);
        for _ in 0..steps {
            let y1 = z[offset];
            let y2 = p[0] * y1 + p[1] * z[offset + stride];
            offset += 2 * stride;
            s1 *= p[2] + p[3] * y1;
            s2 *= p[2] + p[3] * y2;
        }
        *slot = payoff(s1, s2, p[4]);
    }

    /// Each thread averages 200 normal samples using a coalesced layout.
    /// Host-side averaging of these results estimates a+c.
    #[kernel]
    pub fn polynomial(z: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() {
            return;
        }
        let abc = ABC.get();
        let width = thread::blockDim_x() as usize;
        let base = thread::blockIdx_x() as usize * 200 * width + thread::threadIdx_x() as usize;
        let mut sum = 0.0_f32;
        for j in 0..200 {
            let x = z[base + j * width];
            sum += abc[0] * x * x + abc[1] * x + abc[2];
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = sum / 200.0;
        }
    }

    /// Philox4x32-10: a counter-based generator with no global state buffer.
    /// This is a Rust replacement for the device cuRAND API, not its sequence.
    pub fn philox(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
        for _ in 0..10 {
            let a = 0xD2511F53_u64 * c[0] as u64;
            let b = 0xCD9E8D57_u64 * c[2] as u64;
            c = [
                (b >> 32) as u32 ^ c[1] ^ k[0],
                b as u32,
                (a >> 32) as u32 ^ c[3] ^ k[1],
                a as u32,
            ];
            k[0] = k[0].wrapping_add(0x9E3779B9);
            k[1] = k[1].wrapping_add(0xBB67AE85);
        }
        c
    }

    /// Convert two counter words to independent normals using Box-Muller.
    /// The 23-bit midpoint uniforms lie strictly inside (0,1), avoiding log(0).
    pub fn normal_pair(path: u32, step: u32, batch: u32, seed: u64) -> [f32; 2] {
        let c = philox([step, path, batch, 0], [seed as u32, (seed >> 32) as u32]);
        let u1 = ((c[0] >> 9) as f32 + 0.5) * (1.0 / 8388608.0);
        let u2 = ((c[1] >> 9) as f32 + 0.5) * (1.0 / 8388608.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = core::f32::consts::TAU * u2;
        [radius * angle.cos(), radius * angle.sin()]
    }

    /// Occupancy-sized grid with a grid-stride loop, like prac2_device.cu.
    /// Counter-based generation removes RNG_init and state loads/stores.
    #[kernel]
    pub fn pathcalc_device(mut out: DisjointSlice<f32>, seed: u64, batch: u32) {
        let mut i = thread::index_1d().get();
        let stride = thread::gridDim_x() as usize * thread::blockDim_x() as usize;
        let p = PARAMS.get();
        while i < out.len() {
            let mut s1 = 1.0_f32;
            let mut s2 = 1.0_f32;
            for step in 0..N.get() {
                let z = normal_pair(i as u32, step, batch, seed);
                let y2 = p[0] * z[0] + p[1] * z[1];
                s1 *= p[2] + p[3] * z[0];
                s2 *= p[2] + p[3] * y2;
            }
            let value = payoff(s1, s2, p[4]);
            // SAFETY: i < len, and distinct (thread, iteration) pairs own
            // distinct i = global_thread_id + iteration * total_threads.
            unsafe {
                *out.get_unchecked_mut(i) = value;
            }
            i += stride;
        }
    }
}

/// Double-precision reduction of single-precision path payoffs.
fn statistics(values: &[f32]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().map(|&x| x as f64).sum::<f64>() / count;
    let second = values.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / count;
    (mean, ((second - mean * mean).max(0.0) / count).sqrt())
}

/// Scalar reference for one path, using the exact layout of the GPU kernel.
fn reference(z: &[f32], i: usize, steps: usize, width: usize, version: u32, p: &[f32; 5]) -> f32 {
    let block = i / width;
    let lane = i % width;
    let stride = if version == 1 { width } else { 1 };
    let mut ind = block * 2 * steps * width + if version == 1 { lane } else { lane * 2 * steps };
    let (mut s1, mut s2) = (1.0_f32, 1.0_f32);
    for _ in 0..steps {
        let y1 = z[ind];
        ind += stride;
        let y2 = p[0] * y1 + p[1] * z[ind];
        ind += stride;
        s1 *= p[2] + p[3] * y1;
        s2 *= p[2] + p[3] * y2;
    }
    if (s1 - 1.0).abs() < 0.1 && (s2 - 1.0).abs() < 0.1 {
        p[4]
    } else {
        0.0
    }
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const BLOCK_SIZE: u32 = 128;
const SEED: u64 = 1234;
const POLYNOMIAL_SAMPLES: usize = 200;
const HELP: &str = "Usage: prac2_cuda_oxide [buffered|device|polynomial] [paths] [steps] [--verify]\n\
                  defaults: buffered 131072 100; block size 128; seed 1234\n\
                  --verify: CPU comparison; buffered mode only; copies all normals to host";

/// Selects the way each path obtains and uses its normal samples.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Buffered,
    Device,
    Polynomial,
}

/// Holds the validated command-line inputs for a run.
struct Args {
    mode: Mode,
    paths: usize,
    steps: u32,
    verify: bool,
}

/// Holds the grid and padded input sizes for the buffered kernels.
struct BufferPlan {
    launch: LaunchConfig,
    count: usize,
    bytes: usize,
}

/// Owns the loaded kernels, default stream, and reusable timing events.
struct Gpu {
    module: kernels::LoadedModule,
    stream: Arc<CudaStream>,
    start: CudaEvent,
    stop: CudaEvent,
}

impl Gpu {
    /// Load the kernel module and create events with timing enabled.
    fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let stream = ctx.default_stream();
        let module = kernels::load(ctx)?;
        let timing = Some(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT);
        let start = ctx.new_event_with_flags(timing)?;
        let stop = ctx.new_event_with_flags(timing)?;
        Ok(Self {
            module,
            stream,
            start,
            stop,
        })
    }

    /// Upload the step count, payoff parameters, and polynomial coefficients.
    fn upload_constants(&self, steps: u32) -> Result<[f32; 5]> {
        let rho = 0.5_f32;
        let dt = 1.0_f32 / steps as f32;
        let params = [
            rho,
            (1.0 - rho * rho).sqrt(),
            1.0 + 0.05 * dt,
            0.1 * dt.sqrt(),
            (-0.05_f32).exp(),
        ];
        self.module.set_n(&self.stream, &steps)?;
        self.module.set_params(&self.stream, &params)?;
        self.module.set_abc(&self.stream, &[2.0, 3.0, 4.0])?;
        Ok(params)
    }

    /// Time an operation on the default stream after an optional untimed launch.
    fn time(&self, warm_up: bool, mut enqueue: impl FnMut() -> Result<()>) -> Result<f32> {
        if warm_up {
            enqueue()?;
            self.stream.synchronize()?;
        }
        self.start.record(&self.stream)?;
        enqueue()?;
        self.stop.record(&self.stream)?;
        self.stop.synchronize()?;
        Ok(self.start.elapsed_ms(&self.stop)?)
    }
}

/// Map a mode name to its simulation and reject unknown names.
fn parse_mode(name: &str) -> Result<Mode> {
    match name {
        "buffered" => Ok(Mode::Buffered),
        "device" => Ok(Mode::Device),
        "polynomial" => Ok(Mode::Polynomial),
        _ => Err("mode must be buffered, device, or polynomial".into()),
    }
}

/// Collect positional arguments, validate the mode, and parse the path dimensions.
fn parse_args(args: &[String]) -> Result<Args> {
    let verify = args.iter().any(|arg| arg == "--verify");
    let positional: Vec<_> = args
        .iter()
        .map(String::as_str)
        .filter(|&arg| arg != "--verify")
        .collect();
    if positional.len() > 3 {
        return Err("too many arguments; use --help".into());
    }
    let mode = parse_mode(positional.first().copied().unwrap_or("buffered"))?;
    if verify && mode != Mode::Buffered {
        return Err("--verify requires buffered mode".into());
    }
    let paths = positional.get(1).unwrap_or(&"131072").parse::<usize>()?;
    let steps = positional.get(2).unwrap_or(&"100").parse::<u32>()?;
    if paths == 0 || paths > u32::MAX as usize || steps == 0 {
        return Err("paths must fit a nonzero u32; steps must be a nonzero u32".into());
    }
    Ok(Args {
        mode,
        paths,
        steps,
        verify,
    })
}

/// Build a one-dimensional launch with the program's fixed block width.
fn launch_config(blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Size the launch and normal buffer, padding to a whole number of blocks.
fn plan_buffer(args: &Args) -> Result<BufferPlan> {
    let width = BLOCK_SIZE as usize;
    let blocks = args.paths.div_ceil(width);
    let padded = blocks.checked_mul(width).ok_or("path padding overflow")?;
    let samples_per_path = if args.mode == Mode::Polynomial {
        POLYNOMIAL_SAMPLES
    } else {
        (args.steps as usize)
            .checked_mul(2)
            .ok_or("sample count overflow")?
    };
    let count = padded
        .checked_mul(samples_per_path)
        .ok_or("sample count overflow")?;
    let bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or("byte count overflow")?;
    Ok(BufferPlan {
        launch: launch_config(blocks as u32),
        count,
        bytes,
    })
}

/// Pick an occupancy-based grid for the device Philox kernel.
fn device_launch(ctx: &CudaContext, module: &kernels::LoadedModule) -> Result<LaunchConfig> {
    let function = module.as_cuda_module().load_function(
        <kernels::__pathcalc_device_CudaKernel as cuda_host::CudaKernel>::PTX_NAME,
    )?;
    let resident = function.max_active_blocks_per_multiprocessor(BLOCK_SIZE, 0)?;
    let sms = ctx.multiprocessor_count()?;
    let grid = resident.checked_mul(sms).ok_or("occupancy grid overflow")?;
    if grid == 0 {
        return Err("occupancy API returned an empty grid".into());
    }
    println!("SMs: {sms}; active blocks/SM: {resident}; grid blocks: {grid}");
    Ok(launch_config(grid))
}

/// Time the device Philox kernel and report payoff statistics and throughput.
fn run_device(
    ctx: &CudaContext,
    gpu: &Gpu,
    args: &Args,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let config = device_launch(ctx, &gpu.module)?;
    let ms = gpu.time(true, || {
        // SAFETY: The 1D grid-stride loop gives each output index a unique writer.
        unsafe {
            gpu.module
                .pathcalc_device(&gpu.stream, config, out, SEED, 0)?
        };
        Ok(())
    })?;
    let (mean, se) = statistics(&out.to_host_vec(&gpu.stream)?);
    println!("device: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}");
    let samples = 2.0 * args.steps as f64 * args.paths as f64;
    println!("normal samples/s: {:.3e}", samples / (ms as f64 * 1e-3));
    Ok(())
}

/// Allocate the padded buffer and fill it with standard normals using cuRAND.
fn generate_normals(gpu: &Gpu, plan: &BufferPlan) -> Result<DeviceBuffer<f32>> {
    let (count, bytes) = (plan.count, plan.bytes);
    println!(
        "Normal buffer: {count} floats, {:.3} GB (including padded lanes)",
        bytes as f64 / 1e9
    );
    let z = DeviceBuffer::<f32>::zeroed(&gpu.stream, count)?;
    // SAFETY: A live default stream owns the allocation and generator work.
    let rng = unsafe { RNG::new_on_stream(Some(SEED), &gpu.stream) };
    let ms = gpu.time(false, || {
        // SAFETY: count is even and z has count floats on the same stream.
        unsafe { rng.generate_normal_f32(z.cu_deviceptr(), count, 0.0, 1.0) };
        Ok(())
    })?;
    println!("cuRAND: {ms:.3} ms (includes first-call setup)");
    Ok(z)
}

/// Time the polynomial kernel and report the estimated and analytical moments.
fn run_polynomial(
    gpu: &Gpu,
    config: LaunchConfig,
    z: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
) -> Result<()> {
    let ms = gpu.time(true, || {
        // SAFETY: The padded input covers 200 reads per active thread in a 1D grid.
        unsafe { gpu.module.polynomial(&gpu.stream, config, z, out)? };
        Ok(())
    })?;
    let (mean, se) = statistics(&out.to_host_vec(&gpu.stream)?);
    println!("polynomial: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}; expected 6");
    let sigma_mean = (17.0 / (POLYNOMIAL_SAMPLES as f64 * out.len() as f64)).sqrt();
    println!("analytical standard error: {sigma_mean:.8}");
    Ok(())
}

/// Time one buffered layout and return the payoffs after reporting its statistics.
fn run_layout(
    gpu: &Gpu,
    args: &Args,
    config: LaunchConfig,
    z: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<f32>,
    version: u32,
) -> Result<Vec<f32>> {
    let ms = gpu.time(true, || {
        // SAFETY: Padded z covers both layouts, and each active lane owns its output.
        unsafe { gpu.module.pathcalc(&gpu.stream, config, z, out, version)? };
        Ok(())
    })?;
    let values = out.to_host_vec(&gpu.stream)?;
    let (mean, se) = statistics(&values);
    let traffic = (2.0 * args.steps as f64 + 1.0) * args.paths as f64 * 4.0;
    let bandwidth = traffic / (ms as f64 * 1e6);
    println!(
        "version {version}: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}; effective {bandwidth:.2} GB/s"
    );
    Ok(values)
}

/// Compare each GPU payoff with the CPU reference for the selected layout.
fn verify_payoffs(values: &[f32], z: &[f32], steps: u32, version: u32, p: &[f32; 5]) -> Result<()> {
    let mismatches = values
        .iter()
        .enumerate()
        .filter(|&(i, &v)| {
            (v - reference(z, i, steps as usize, BLOCK_SIZE as usize, version, p)).abs() > 1e-6
        })
        .count();
    if mismatches != 0 {
        return Err(format!("version {version}: {mismatches} CPU/GPU payoff mismatches; retry build --no-fmad to exclude FMA boundary effects").into());
    }
    println!("CPU reference: all {} payoffs matched", values.len());
    Ok(())
}

/// Generate normals once, then dispatch to the polynomial kernel or both layouts.
fn run_buffered(
    gpu: &Gpu,
    args: &Args,
    plan: &BufferPlan,
    out: &mut DeviceBuffer<f32>,
    params: &[f32; 5],
) -> Result<()> {
    let z = generate_normals(gpu, plan)?;
    if args.mode == Mode::Polynomial {
        return run_polynomial(gpu, plan.launch, &z, out);
    }
    let host_z = args
        .verify
        .then(|| z.to_host_vec(&gpu.stream))
        .transpose()?;
    for version in [1, 2] {
        let values = run_layout(gpu, args, plan.launch, &z, out, version)?;
        if let Some(normals) = &host_z {
            verify_payoffs(&values, normals, args.steps, version, params)?;
        }
    }
    Ok(())
}

/// Parse the command, initialize CUDA device zero, and dispatch the simulation.
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help") {
        println!("{HELP}");
        return Ok(());
    }
    let args = parse_args(&args)?;
    let plan = plan_buffer(&args)?;
    let ctx = CudaContext::new(0)?;
    let gpu = Gpu::new(&ctx)?;
    let params = gpu.upload_constants(args.steps)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&gpu.stream, args.paths)?;
    match args.mode {
        Mode::Device => run_device(&ctx, &gpu, &args, &mut out),
        Mode::Buffered | Mode::Polynomial => run_buffered(&gpu, &args, &plan, &mut out, &params),
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// Random123's published Philox4x32-10 zero-key/zero-counter test vector.
    #[test]
    fn philox_known_answer() {
        assert_eq!(
            kernels::philox([0; 4], [0; 2]),
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        );
    }

    /// Verify both layouts are permutations of a padded input allocation.
    /// Covers one block, multiple blocks, and a partially occupied final block.
    #[test]
    fn layouts_cover_all_samples_without_overlap() {
        for paths in [1_usize, 32, 127, 128, 129, 256] {
            for width in [32_usize, 128] {
                let steps = 3;
                let padded = paths.div_ceil(width) * width;
                for version in [1, 2] {
                    let mut seen = vec![false; padded * 2 * steps];
                    for i in 0..padded {
                        let stride = if version == 1 { width } else { 1 };
                        let base = (i / width) * 2 * steps * width
                            + if version == 1 {
                                i % width
                            } else {
                                (i % width) * 2 * steps
                            };
                        for j in 0..2 * steps {
                            let index = base + j * stride;
                            assert!(!seen[index]);
                            seen[index] = true;
                        }
                    }
                    assert!(seen.iter().all(|&x| x));
                }
            }
        }
    }

    /// A deterministic smoke check of normal means, variances, and covariance.
    /// This is not a general statistical certification of the generator.
    #[test]
    fn normals_have_expected_moments() {
        let mut sum = [0.0_f64; 5];
        let count = 100_000;
        for i in 0..count {
            let z = kernels::normal_pair(i, 0, 0, 1234);
            let (x, y) = (z[0] as f64, z[1] as f64);
            sum[0] += x;
            sum[1] += y;
            sum[2] += x * x;
            sum[3] += y * y;
            sum[4] += x * y;
        }
        let n = count as f64;
        assert!((sum[0] / n).abs() < 0.02 && (sum[1] / n).abs() < 0.02);
        assert!((sum[2] / n - 1.0).abs() < 0.03 && (sum[3] / n - 1.0).abs() < 0.03);
        assert!((sum[4] / n).abs() < 0.02);
    }
}
