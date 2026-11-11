use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_core::curand::RNG;
use cuda_device::{ConstantMemory, DisjointSlice, constant, cuda_module, kernel, thread};

/// kernels share host-populated constant memory
#[cuda_module]
mod kernels {
    use super::*;

    /// #Euler steps in each simulated path.
    #[constant]
    static N: ConstantMemory<u32> = ConstantMemory::UNINIT;
    /// [rho, alpha, 1+r*dt, sigma*sqrt(dt), exp(-r*T)].
    #[constant]
    static PARAMS: ConstantMemory<[f32; 5]> = ConstantMemory::UNINIT;
    /// Polynomial coefficients [a, b, c]
    #[constant]
    static ABC: ConstantMemory<[f32; 3]> = ConstantMemory::UNINIT;

    /// versions 1 and 2: selected by a uniform scalar argument.
    /// version 1 reads adjacent elements across lanes; version 2 across time.
    #[kernel]
    pub fn pathcalc(z: &[f32], mut out: DisjointSlice<f32>, version: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() { return; }
        let nsteps = N.get() as usize;
        let p = PARAMS.get();
        let width = thread::blockDim_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let block = thread::blockIdx_x() as usize;
        let stride = if version == 1 { width } else { 1 };
        let mut ind = block * (2 * nsteps * width)
            + if version == 1 { lane } else { lane * 2 * nsteps };
        let mut s1 = 1.0_f32;
        let mut s2 = 1.0_f32;
        for _ in 0..nsteps {
            let y1 = z[ind];
            ind += stride;
            let y2 = p[0] * y1 + p[1] * z[ind];
            ind += stride;
            s1 *= p[2] + p[3] * y1;
            s2 *= p[2] + p[3] * y2;
        }
        let value = if (s1 - 1.0).abs() < 0.1 && (s2 - 1.0).abs() < 0.1 {
            p[4]
        } else { 0.0 };
        if let Some(slot) = out.get_mut(idx) { *slot = value; }
    }

    /// each thread averages 200 normal samples using a coalesced layout.
    /// host-side averaging of these results estimates a+c.
    #[kernel]
    pub fn polynomial(z: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= out.len() { return; }
        let abc = ABC.get();
        let width = thread::blockDim_x() as usize;
        let base = thread::blockIdx_x() as usize * 200 * width
            + thread::threadIdx_x() as usize;
        let mut sum = 0.0_f32;
        for j in 0..200 {
            let x = z[base + j * width];
            sum += abc[0] * x * x + abc[1] * x + abc[2];
        }
        if let Some(slot) = out.get_mut(idx) { *slot = sum / 200.0; }
    }

    /// Philox4x32-10: a counter-based generator with no global state buffer.
    /// this is a Rust replacement for the device cuRAND API, not its sequence.
    pub fn philox(mut c: [u32; 4], mut k: [u32; 2]) -> [u32; 4] {
        for _ in 0..10 {
            let a = 0xD2511F53_u64 * c[0] as u64;
            let b = 0xCD9E8D57_u64 * c[2] as u64;
            c = [(b >> 32) as u32 ^ c[1] ^ k[0], b as u32,
                 (a >> 32) as u32 ^ c[3] ^ k[1], a as u32];
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
            let value = if (s1 - 1.0).abs() < 0.1 && (s2 - 1.0).abs() < 0.1 {
                p[4]
            } else { 0.0 };
            // SAFETY: i < len, and distinct (thread, iteration) pairs own
            // distinct i = global_thread_id + iteration * total_threads.
            unsafe { *out.get_unchecked_mut(i) = value; }
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
fn reference(z: &[f32], i: usize, steps: usize, width: usize, version: u32,
             p: &[f32; 5]) -> f32 {
    let block = i / width;
    let lane = i % width;
    let stride = if version == 1 { width } else { 1 };
    let mut ind = block * 2 * steps * width
        + if version == 1 { lane } else { lane * 2 * steps };
    let (mut s1, mut s2) = (1.0_f32, 1.0_f32);
    for _ in 0..steps {
        let y1 = z[ind];
        ind += stride;
        let y2 = p[0] * y1 + p[1] * z[ind];
        ind += stride;
        s1 *= p[2] + p[3] * y1;
        s2 *= p[2] + p[3] * y2;
    }
    if (s1 - 1.0).abs() < 0.1 && (s2 - 1.0).abs() < 0.1 { p[4] } else { 0.0 }
}

/// Command-line driver. See README for memory requirements and examples.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|s| s == "--help") {
        println!("Usage: prac2_cuda_oxide [buffered|device|polynomial] [paths] [steps] [--verify]\n\
                  defaults: buffered 131072 100; block size 128; seed 1234\n\
                  --verify: CPU comparison; buffered mode only; copies all normals to host");
        return Ok(());
    }
    let verify = args.iter().any(|s| s == "--verify");
    let positional: Vec<&str> = args.iter().filter(|s| s.as_str() != "--verify")
        .map(String::as_str).collect();
    if positional.len() > 3 { return Err("too many arguments; use --help".into()); }
    let mode = positional.first().copied().unwrap_or("buffered");
    if !["buffered", "device", "polynomial"].contains(&mode) {
        return Err("mode must be buffered, device, or polynomial".into());
    }
    if verify && mode != "buffered" { return Err("--verify requires buffered mode".into()); }
    let paths: usize = positional.get(1).unwrap_or(&"131072").parse()?;
    let steps: u32 = positional.get(2).unwrap_or(&"100").parse()?;
    if paths == 0 || paths > u32::MAX as usize || steps == 0 {
        return Err("paths must fit a nonzero u32; steps must be a nonzero u32".into());
    }
    let blocks = paths.div_ceil(128);
    let padded = blocks.checked_mul(128).ok_or("path padding overflow")?;
    let samples_per_path = if mode == "polynomial" { 200 } else {
        (steps as usize).checked_mul(2).ok_or("sample count overflow")?
    };
    let count = padded.checked_mul(samples_per_path).ok_or("sample count overflow")?;
    let bytes = count.checked_mul(4).ok_or("byte count overflow")?;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let rho = 0.5_f32;
    let dt = 1.0_f32 / steps as f32; // T = 1, as in Giles's source.
    let p = [rho, (1.0 - rho * rho).sqrt(), 1.0 + 0.05 * dt,
             0.1 * dt.sqrt(), (-0.05_f32).exp()];
    module.set_n(&stream, &steps)?;
    module.set_params(&stream, &p)?;
    let abc = [2.0_f32, 3.0, 4.0];
    module.set_abc(&stream, &abc)?;
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, paths)?;
    // The default event constructor disables timing; request timed events.
    let start = ctx.new_event_with_flags(Some(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let stop = ctx.new_event_with_flags(Some(cuda_bindings::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let config = LaunchConfig { grid_dim: (blocks as u32, 1, 1),
        block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    if mode == "device" {
        // Use the macro's marker instead of hard-coding a mangled PTX symbol.
        let function = module.as_cuda_module().load_function(
            <kernels::__pathcalc_device_CudaKernel as cuda_host::CudaKernel>::PTX_NAME)?;
        let resident = function.max_active_blocks_per_multiprocessor(128, 0)?;
        let sms = ctx.multiprocessor_count()?;
        let grid = resident.checked_mul(sms).ok_or("occupancy grid overflow")?;
        if grid == 0 { return Err("occupancy API returned an empty grid".into()); }
        println!("SMs: {sms}; active blocks/SM: {resident}; grid blocks: {grid}");
        let config = LaunchConfig { grid_dim: (grid, 1, 1), ..config };
        // SAFETY: 1D grid, 128 threads/block, exclusive output buffer; each
        // thread owns its grid-stride indices. No RNG state allocation needed.
        unsafe { module.pathcalc_device(&stream, config, &mut out, 1234_u64, 0_u32)?; }
        stream.synchronize()?; // Warm up before the measured launch.
        start.record(&stream)?;
        unsafe { module.pathcalc_device(&stream, config, &mut out, 1234_u64, 0_u32)?; }
        stop.record(&stream)?;
        stop.synchronize()?;
        let ms = start.elapsed_ms(&stop)?;
        let (mean, se) = statistics(&out.to_host_vec(&stream)?);
        println!("device: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}");
        println!("normal samples/s: {:.3e}", 2.0 * steps as f64 * paths as f64 / (ms as f64 * 1e-3));
    } else {
        println!("Normal buffer: {count} floats, {:.3} GB (including padded lanes)", bytes as f64 / 1e9);
        let z = DeviceBuffer::<f32>::zeroed(&stream, count)?;
        // SAFETY: live CUDA context and stream; count is even and z contains
        // count floats. All allocation, RNG, and kernel work uses this stream.
        let rng = unsafe { RNG::new_on_stream(Some(1234), &stream) };
        start.record(&stream)?;
        unsafe { rng.generate_normal_f32(z.cu_deviceptr(), count, 0.0, 1.0); }
        stop.record(&stream)?;
        stop.synchronize()?;
        println!("cuRAND: {:.3} ms (includes first-call setup)", start.elapsed_ms(&stop)?);
        if mode == "polynomial" {
            // SAFETY: padded input covers every active thread's 200 reads;
            // 1D launch and one unique output per active thread.
            unsafe { module.polynomial(&stream, config, &z, &mut out)?; }
            stream.synchronize()?;
            start.record(&stream)?;
            unsafe { module.polynomial(&stream, config, &z, &mut out)?; }
            stop.record(&stream)?;
            stop.synchronize()?;
            let ms = start.elapsed_ms(&stop)?;
            let (mean, se) = statistics(&out.to_host_vec(&stream)?);
            println!("polynomial: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}; expected 6");
            let sigma_mean = ((2.0 * 2.0_f64.powi(2) + 3.0_f64.powi(2)) / (200.0 * paths as f64)).sqrt();
            println!("analytical standard error: {sigma_mean:.8}");
        } else {
            let host_z = if verify { Some(z.to_host_vec(&stream)?) } else { None };
            for version in [1_u32, 2] {
                // SAFETY: 1D grid; padded z covers both indexing formulas;
                // each active thread owns exactly one element of out.
                unsafe { module.pathcalc(&stream, config, &z, &mut out, version)?; }
                stream.synchronize()?;
                start.record(&stream)?;
                unsafe { module.pathcalc(&stream, config, &z, &mut out, version)?; }
                stop.record(&stream)?;
                stop.synchronize()?;
                let ms = start.elapsed_ms(&stop)?;
                let values = out.to_host_vec(&stream)?;
                let (mean, se) = statistics(&values);
                let traffic = (2.0 * steps as f64 + 1.0) * paths as f64 * 4.0;
                println!("version {version}: {ms:.3} ms; mean {mean:.8}; standard error {se:.8}; effective {:.2} GB/s",
                    traffic / (ms as f64 * 1e6));
                if let Some(z) = &host_z {
                    let mismatches = values.iter().enumerate().filter(|&(i, &v)|
                        (v - reference(z, i, steps as usize, 128, version, &p)).abs() > 1e-6).count();
                    if mismatches != 0 {
                        return Err(format!("version {version}: {mismatches} CPU/GPU payoff mismatches; retry build --no-fmad to exclude FMA boundary effects").into());
                    }
                    println!("CPU reference: all {paths} payoffs matched");
                }
            }
        }
        // RNG's Drop destroys the generator after its stream work completed.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random123's published Philox4x32-10 zero-key/zero-counter test vector.
    #[test]
    fn philox_known_answer() {
        assert_eq!(kernels::philox([0; 4], [0; 2]),
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
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
                            + if version == 1 { i % width } else { (i % width) * 2 * steps };
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
            sum[0] += x; sum[1] += y;
            sum[2] += x*x; sum[3] += y*y; sum[4] += x*y;
        }
        let n = count as f64;
        assert!((sum[0]/n).abs() < 0.02 && (sum[1]/n).abs() < 0.02);
        assert!((sum[2]/n - 1.0).abs() < 0.03 && (sum[3]/n - 1.0).abs() < 0.03);
        assert!((sum[4]/n).abs() < 0.02);
    }
}
