use super::{cpu::Storage, *};
use crate::{cfl, finite};

/// Check a computed real buffer for finite values and convert it to the selected CPU storage type.
pub(super) fn real<A: Storage<f64>>(v: Vec<f64>) -> Result<A> {
    finite(&v)?;
    Ok(v.into())
}
/// Return the preceding index on a periodic line.
pub(super) fn left(i: usize, n: usize) -> usize {
    if i == 0 {
        n - 1
    } else {
        i - 1
    }
}
/// Return the following index on a periodic line.
pub(super) fn right(i: usize, n: usize) -> usize {
    if i + 1 == n {
        0
    } else {
        i + 1
    }
}
/// Evaluate the physical flux f(u) at a single state.
fn physical(f: Flux, u: f64) -> f64 {
    match f {
        Flux::Linear(a) => a * u,
        Flux::Burgers => 0.5 * u * u,
    }
}
/// Evaluate the flux at the interface of the scalar Riemann problem. The Burgers rarefaction can have zero interface speed.
fn godunov(f: Flux, l: f64, r: f64) -> f64 {
    match f {
        Flux::Linear(a) => a * if a >= 0. { l } else { r },
        Flux::Burgers if l <= r => 0.5 * l.max(0.).max(-r).powi(2),
        Flux::Burgers => 0.5 * l.abs().max(r.abs()).powi(2),
    }
}
/// Build the numerical flux for a pair of adjacent states. Lax–Wendroff evaluates the physical flux on a predicted half-time state.
fn interface(f: Flux, s: Scheme, l: f64, r: f64, ratio: f64) -> f64 {
    let (fl, fr) = (physical(f, l), physical(f, r));
    match s {
        Scheme::Ftcs => 0.5 * (fl + fr),
        Scheme::Lax => 0.5 * (fl + fr - (r - l) / ratio),
        Scheme::LaxWendroff => physical(f, 0.5 * (l + r - ratio * (fr - fl))),
        _ => godunov(f, l, r),
    }
}
/// Limit a piecewise linear reconstruction by the minmod rule.
fn slope(a: f64, b: f64) -> f64 {
    if a.signum() == b.signum() {
        a.signum() * a.abs().min(b.abs())
    } else {
        0.
    }
}
/// Reconstruct both sides of each interface and evaluate their Godunov flux.
fn muscl_faces(u: &[f64], f: Flux) -> Vec<f64> {
    let n = u.len();
    let m: Vec<_> = (0..n)
        .map(|i| slope(u[i] - u[left(i, n)], u[right(i, n)] - u[i]))
        .collect();
    (0..n)
        .map(|i| {
            let j = right(i, n);
            godunov(f, u[i] + 0.5 * m[i], u[j] - 0.5 * m[j])
        })
        .collect()
}
/// Subtract adjacent interface fluxes so the same interface contributes with opposite signs to its two neighbors.
fn divergence(u: &[f64], faces: &[f64], ratio: f64) -> Vec<f64> {
    (0..u.len())
        .map(|i| u[i] - ratio * (faces[i] - faces[left(i, u.len())]))
        .collect()
}
/// Advance one periodic scalar conservation step. MUSCL reconstructs twice and combines the stages with SSPRK2.
pub(super) fn flux<A: Storage<f64>>(u: &A, f: Flux, s: Scheme, ratio: f64) -> Result<A> {
    let u = u.as_ref();
    if u.len() < 3 {
        return Err(Error::Argument);
    }
    let n = u.len();
    if matches!(s, Scheme::Muscl) {
        let stage = divergence(u, &muscl_faces(u, f), ratio);
        let step = divergence(&stage, &muscl_faces(&stage, f), ratio);
        return real(u.iter().zip(step).map(|(a, b)| 0.5 * (a + b)).collect());
    }
    let faces: Vec<_> = (0..n)
        .map(|i| interface(f, s, u[i], u[right(i, n)], ratio))
        .collect();
    real(divergence(u, &faces, ratio))
}
/// Use the centered spatial difference at the current time to update the previous-time state.
pub(super) fn leapfrog<A: Storage<f64>>(prev: &A, u: &A, c: f64) -> Result<A> {
    let n = u.len();
    if n < 3 || prev.len() != n {
        return Err(Error::Argument);
    }
    let (p, u) = (prev.as_ref(), u.as_ref());
    real(
        (0..n)
            .map(|i| p[i] - c * (u[right(i, n)] - u[left(i, n)]))
            .collect(),
    )
}
/// Update face-centered q first, then use the new q to update node-centered p.
pub(super) fn wave<A: Storage<f64>>(p: &A, q: &A, c: f64) -> Result<(A, A)> {
    let n = p.len();
    if n < 3 || q.len() != n {
        return Err(Error::Argument);
    }
    let (p, q) = (p.as_ref(), q.as_ref());
    let v: Vec<_> = (0..n).map(|i| q[i] - c * (p[right(i, n)] - p[i])).collect();
    let u = (0..n).map(|i| p[i] - c * (v[i] - v[left(i, n)])).collect();
    Ok((real(u)?, real(v)?))
}
/// Combine a four-neighbor average with centered transport differences along both axes.
pub(super) fn lax2<A: Storage<f64>>(u: &A, g: Grid, cx: f64, cy: f64) -> Result<A> {
    g.check(u)?;
    if g.ny < 3 {
        return Err(Error::Argument);
    }
    let u = u.as_ref();
    real(
        (0..u.len())
            .map(|i| {
                let (x, y) = (i % g.nx, i / g.nx);
                let (l, r) = (u[y * g.nx + left(x, g.nx)], u[y * g.nx + right(x, g.nx)]);
                let (d, t) = (u[left(y, g.ny) * g.nx + x], u[right(y, g.ny) * g.nx + x]);
                0.25 * (l + r + d + t) - 0.5 * (cx * (r - l) + cy * (t - d))
            })
            .collect(),
    )
}
/// Apply periodic Lax–Wendroff to each line in the chosen direction, with a separate constant speed and CFL check for each line.
pub(super) fn sweep<A: Storage<f64>>(
    u: &A,
    g: Grid,
    axis: Axis,
    speed: &A,
    ratio: f64,
) -> Result<A> {
    g.check(u)?;
    let (n, lanes) = dimensions(g, axis);
    if n < 3 || speed.len() != lanes {
        return Err(Error::Argument);
    }
    let (u, speed) = (u.as_ref(), speed.as_ref());
    let mut v = vec![0.; u.len()];
    for (lane, a) in speed.iter().enumerate() {
        let c = a * ratio;
        cfl(c.abs())?;
        for i in 0..n {
            let j = cell(g, axis, lane, i);
            let l = u[cell(g, axis, lane, left(i, n))];
            let r = u[cell(g, axis, lane, right(i, n))];
            v[j] = u[j] - 0.5 * c * (r - l) + 0.5 * c * c * (r - 2. * u[j] + l);
        }
    }
    real(v)
}
/// Return (points per line, number of independent lines) for the selected sweep axis.
pub(super) fn dimensions(g: Grid, axis: Axis) -> (usize, usize) {
    match axis {
        Axis::X => (g.nx, g.ny),
        Axis::Y => (g.ny, g.nx),
    }
}
/// Map a line and its local index to the row-major flat index.
pub(super) fn cell(g: Grid, axis: Axis, lane: usize, i: usize) -> usize {
    match axis {
        Axis::X => lane * g.nx + i,
        Axis::Y => i * g.nx + lane,
    }
}
/// Sum velocity rows with quadrature weight dv to produce one density per x point.
pub(super) fn integrate_v<A: Storage<f64>>(u: &A, g: Grid, dv: f64) -> Result<A> {
    g.check(u)?;
    let mut rho = vec![0.; g.nx];
    for row in u.as_ref().chunks_exact(g.nx) {
        for (a, b) in rho.iter_mut().zip(row) {
            *a += dv * b;
        }
    }
    real(rho)
}
