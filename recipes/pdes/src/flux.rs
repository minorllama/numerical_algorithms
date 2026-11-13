//! Periodic flux-conservative stencils and the staggered wave update (§20.1).
use crate::{
    cfl,
    matrix::{Flux, Scheme},
    positive, Backend, Error, Result,
};

/// One scalar conservation-law step. FTCS is included for stability experiments.
///
/// Use a periodic buffer with at least three entries. Pass physical dt and dx, and call again with the returned buffer for each subsequent step.
pub fn step<B: Backend>(
    b: &B,
    u: &B::Real,
    f: Flux,
    s: Scheme,
    dt: f64,
    dx: f64,
) -> Result<B::Real> {
    positive(&[dt, dx])?;
    let speed = match f {
        Flux::Linear(a) => a.abs(),
        Flux::Burgers => b.max_abs(u)?,
    };
    if matches!(s, Scheme::Upwind) && !matches!(f, Flux::Linear(_)) {
        return Err(Error::Argument);
    }
    let limit = if matches!(s, Scheme::Muscl) { 0.5 } else { 1. };
    cfl(speed * (dt / dx) / limit)?;
    b.flux(u, f, s, dt / dx)
}

/// Centered leapfrog for constant-speed advection. Supply a consistent first step.
///
/// The two inputs represent times n−1 and n. The returned buffer represents n+1. Pass c = speed * dt / dx.
pub fn leapfrog<B: Backend>(b: &B, previous: &B::Real, u: &B::Real, c: f64) -> Result<B::Real> {
    cfl(c.abs())?;
    b.leapfrog(previous, u, c)
}

/// Staggered update for p_t=-a q_x, q_t=-a p_x, with c=a dt/dx.
/// p is at integer times and nodes, q at the preceding half time and right faces.
///
/// Returns (p at the next integer time, q at the next half time). Both input buffers are periodic and have equal length.
pub fn wave<B: Backend>(b: &B, p: &B::Real, q: &B::Real, c: f64) -> Result<(B::Real, B::Real)> {
    cfl(c.abs())?;
    b.wave(p, q, c)
}
