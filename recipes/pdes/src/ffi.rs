//! C ABI for caller-owned, contiguous f64 buffers. See include/pde_recipes.h.
use crate::{
    diffusion::{black_scholes, Call},
    elliptic,
    matrix::{Boundary, Scheme},
    multidim::{adi_step, burgers, vlasov, PhaseSpace},
    Cpu, Error, Grid, Result,
};
use std::{mem::size_of, panic::catch_unwind, ptr, slice};

/// Check pointer alignment and element-count bounds before constructing a slice. The caller guarantees the underlying allocation.
fn valid(p: *const f64, n: usize) -> Result<()> {
    if p.is_null() || !p.is_aligned() || n == 0 || n > isize::MAX as usize / size_of::<f64>() {
        Err(Error::Argument)
    } else {
        Ok(())
    }
}

/// Copy the input, run the solver and commit to the output buffer only on success. Translate Rust errors and panics into ABI status codes.
fn run<F>(input: *const f64, out: *mut f64, n: usize, solve: F) -> i32
where
    F: FnOnce(Vec<f64>) -> Result<Vec<f64>> + std::panic::UnwindSafe,
{
    let status = catch_unwind(|| {
        valid(input, n)?;
        valid(out, n)?;
        // The ABI caller supplies valid buffers. Own the data before writing.
        let data = unsafe { slice::from_raw_parts(input, n).to_vec() };
        crate::finite(&data)?;
        let value = solve(data)?;
        if value.len() != n {
            return Err(Error::Argument);
        }
        crate::finite(&value)?;
        unsafe {
            ptr::copy_nonoverlapping(value.as_ptr(), out, n);
        }
        Ok(())
    });
    match status {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => e as i32,
        Err(_) => 127,
    }
}

/// Check the grid dimensions and derive the flat buffer length before calling run.
fn run_grid<F>(input: *const f64, out: *mut f64, grid: Grid, f: F) -> i32
where
    F: FnOnce(Vec<f64>) -> Result<Vec<f64>> + std::panic::UnwindSafe,
{
    match grid.size() {
        Ok(n) => run(input, out, n, f),
        Err(e) => e as i32,
    }
}

/// Decode the boundary integer documented in the C header.
fn boundary(code: i32) -> Result<Boundary> {
    match code {
        0 => Ok(Boundary::Periodic),
        1 => Ok(Boundary::Dirichlet),
        2 => Ok(Boundary::Neumann),
        _ => Err(Error::Argument),
    }
}

/// Periodic viscous Burgers. Methods 0, 1 and 2 select LW, MUSCL and Godunov.
/// # Safety
/// input and out must point to n readable and writable f64 values respectively.
#[no_mangle]
pub unsafe extern "C" fn pde_burgers(
    input: *const f64,
    out: *mut f64,
    n: usize,
    dx: f64,
    nu: f64,
    dt: f64,
    steps: usize,
    method: i32,
) -> i32 {
    let scheme = match method {
        0 => Scheme::LaxWendroff,
        1 => Scheme::Muscl,
        2 => Scheme::Godunov,
        _ => return Error::Argument as i32,
    };
    run(input, out, n, |u| {
        burgers(&Cpu::<Vec<f64>>::default(), u, nu, dx, dt, steps, scheme)
    })
}

/// European call price from a supplied payoff on a uniform spot grid.
/// # Safety
/// input and out must point to n readable and writable f64 values respectively.
#[no_mangle]
pub unsafe extern "C" fn pde_black_scholes(
    input: *const f64,
    out: *mut f64,
    n: usize,
    s_max: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    dt: f64,
    steps: usize,
) -> i32 {
    let p = Call {
        s_max,
        strike,
        rate,
        volatility,
    };
    run(input, out, n, |u| {
        black_scholes(&Cpu::<Vec<f64>>::default(), u, p, dt, steps)
    })
}

/// Electrostatic Vlasov–Poisson. Arrays have shape (nv,nx).
/// # Safety
/// input and out must point to nx*nv readable and writable f64 values respectively.
#[no_mangle]
pub unsafe extern "C" fn pde_vlasov(
    input: *const f64,
    out: *mut f64,
    nx: usize,
    nv: usize,
    dx: f64,
    dv: f64,
    v_min: f64,
    dt: f64,
    steps: usize,
) -> i32 {
    let p = PhaseSpace {
        grid: Grid { nx, ny: nv },
        dx,
        dv,
        v_min,
    };
    run_grid(input, out, p.grid, |u| {
        vlasov(&Cpu::<Vec<f64>>::default(), u, p, dt, steps)
    })
}

/// Poisson with axis boundary codes 0 periodic, 1 Dirichlet, 2 Neumann.
/// # Safety
/// input and out must point to nx*ny readable and writable f64 values respectively.
#[no_mangle]
pub unsafe extern "C" fn pde_poisson(
    input: *const f64,
    out: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    bx: i32,
    by: i32,
) -> i32 {
    let g = Grid { nx, ny };
    run_grid(input, out, g, |u| {
        elliptic::poisson(
            &Cpu::<Vec<f64>>::default(),
            &u,
            g,
            [dx, dy],
            [boundary(bx)?, boundary(by)?],
        )
    })
}

/// Copy the two length-ny coefficient buffers after checking their pointers.
unsafe fn coefficient_pair(b: *const f64, c: *const f64, n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    valid(b, n)?;
    valid(c, n)?;
    Ok(unsafe {
        (
            slice::from_raw_parts(b, n).to_vec(),
            slice::from_raw_parts(c, n).to_vec(),
        )
    })
}

/// Separable elliptic cyclic reduction with homogeneous Dirichlet boundaries.
/// # Safety
/// input and out must span nx*ny values. drift and reaction must each span ny
/// readable f64 values. All inputs must remain valid for the call.
#[no_mangle]
pub unsafe extern "C" fn pde_elliptic_cr(
    input: *const f64,
    out: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    drift: *const f64,
    reaction: *const f64,
) -> i32 {
    let g = Grid { nx, ny };
    run_grid(input, out, g, |u| {
        let (b, c) = unsafe { coefficient_pair(drift, reaction, ny)? };
        elliptic::cyclic_reduction(&Cpu::<Vec<f64>>::default(), &u, g, [dx, dy], &b, &c)
    })
}

/// Periodic 2D diffusion with ADI ratios r_x, r_y.
/// # Safety
/// input and out must point to nx*ny readable and writable f64 values respectively.
#[no_mangle]
pub unsafe extern "C" fn pde_adi(
    input: *const f64,
    out: *mut f64,
    nx: usize,
    ny: usize,
    rx: f64,
    ry: f64,
    steps: usize,
) -> i32 {
    let g = Grid { nx, ny };
    run_grid(input, out, g, |mut u| {
        if steps == 0 {
            return Err(Error::Argument);
        }
        let b = Cpu::<Vec<f64>>::default();
        for _ in 0..steps {
            u = adi_step(&b, &u, g, [rx, ry])?;
        }
        Ok(u)
    })
}

/// Direct block cyclic reduction with nx+1 a power of two.
/// # Safety
/// input and out must span nx*ny values. drift and reaction must each span ny
/// readable f64 values. All inputs must remain valid for the call.
#[no_mangle]
pub unsafe extern "C" fn pde_block_cr(
    input: *const f64,
    out: *mut f64,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    drift: *const f64,
    reaction: *const f64,
) -> i32 {
    let g = Grid { nx, ny };
    run_grid(input, out, g, |u| {
        let (b, c) = unsafe { coefficient_pair(drift, reaction, ny)? };
        elliptic::block_cyclic_reduction(&Cpu::<Vec<f64>>::default(), &u, g, [dx, dy], &b, &c)
    })
}
