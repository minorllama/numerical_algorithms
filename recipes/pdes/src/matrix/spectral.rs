use super::{
    cpu::Storage,
    linear::{cr, Equation},
    stencil::{cell, dimensions, real},
    *,
};
use crate::{finite, positive};
use rustfft::FftPlanner;
use std::f64::consts::{PI, TAU};

type Plan = FftPlanner<f64>;
type Z = Complex64;
const ZERO: Z = Z::new(0., 0.);

/// Transform a complex line in place. Divide by its length on the inverse pass to undo RustFFT's unnormalized transform.
fn fft(p: &mut Plan, x: &mut [Z], inv: bool) {
    let n = x.len();
    if inv {
        p.plan_fft_inverse(n).process(x);
        x.iter_mut().for_each(|z| *z /= n as f64);
    } else {
        p.plan_fft_forward(n).process(x);
    }
}
/// Use an odd extension to apply the sine transform for interior Dirichlet nodes.
fn sine(p: &mut Plan, x: &mut [Z], inv: bool) {
    let n = x.len();
    let m = 2 * (n + 1);
    let mut y = vec![ZERO; m];
    for (j, &z) in x.iter().enumerate() {
        y[j + 1] = z;
        y[m - j - 1] = -z;
    }
    fft(p, &mut y, false);
    let weight = if inv { 1. / (n + 1) as f64 } else { 0.5 };
    for (j, z) in x.iter_mut().enumerate() {
        *z = Z::i() * weight * y[j + 1];
    }
}
/// Return the half-cell phase factor for the cosine transform.
fn phase(k: usize, n: usize) -> Z {
    Z::from_polar(1., PI * k as f64 / (2 * n) as f64)
}
/// Use an even reflection and phase correction to transform a cell-centered Neumann line.
fn cosine(p: &mut Plan, x: &mut [Z], inv: bool) {
    let n = x.len();
    let m = 2 * n;
    let mut y = vec![ZERO; m];
    if inv {
        y[0] = 2. * x[0];
        for k in 1..n {
            let a = phase(k, n);
            y[k] = 2. * a * x[k];
            y[m - k] = 2. * a.conj() * x[k];
        }
        fft(p, &mut y, true);
        x.copy_from_slice(&y[..n]);
    } else {
        for (j, &z) in x.iter().enumerate() {
            y[j] = z;
            y[m - 1 - j] = z;
        }
        fft(p, &mut y, false);
        for (k, z) in x.iter_mut().enumerate() {
            *z = 0.5 * phase(k, n).conj() * y[k];
        }
    }
}
/// Select the periodic, Dirichlet or Neumann transform for one axis.
fn transform(p: &mut Plan, x: &mut [Z], b: Boundary, inv: bool) {
    match b {
        Boundary::Periodic => fft(p, x, inv),
        Boundary::Dirichlet => sine(p, x, inv),
        Boundary::Neumann => cosine(p, x, inv),
    }
}
/// Gather, transform and scatter each independent line along the selected axis.
fn axis(p: &mut Plan, u: &mut [Z], g: Grid, a: Axis, b: Boundary, inv: bool) {
    let (n, lanes) = dimensions(g, a);
    let mut row = vec![ZERO; n];
    for j in 0..lanes {
        for (i, z) in row.iter_mut().enumerate() {
            *z = u[cell(g, a, j, i)];
        }
        transform(p, &mut row, b, inv);
        for (i, &z) in row.iter().enumerate() {
            u[cell(g, a, j, i)] = z;
        }
    }
}
/// Return the finite difference Laplacian eigenvalue for one mode and boundary convention.
fn eigen(k: usize, n: usize, h: f64, bc: Boundary) -> f64 {
    let theta = match bc {
        Boundary::Periodic => PI * k.min(n - k) as f64 / n as f64,
        Boundary::Dirichlet => 0.5 * PI * (k + 1) as f64 / (n + 1) as f64,
        Boundary::Neumann => 0.5 * PI * k as f64 / n as f64,
    };
    -4. * (theta.sin() / h).powi(2)
}
/// Divide by the sum of the two discrete axis eigenvalues and fix the constant mode to zero when present.
fn divide(v: &mut [Z], g: Grid, h: [f64; 2], bc: [Boundary; 2]) {
    for (i, z) in v.iter_mut().enumerate() {
        let d = eigen(i % g.nx, g.nx, h[0], bc[0]) + eigen(i / g.nx, g.ny, h[1], bc[1]);
        *z = if d == 0. { ZERO } else { *z / d };
    }
}
/// Check the zero-mean compatibility condition when needed, solve mode by mode and transform back to the real grid.
pub(super) fn poisson<A: Storage<f64>>(
    rhs: &A,
    g: Grid,
    h: [f64; 2],
    bc: [Boundary; 2],
) -> Result<A> {
    g.check(rhs)?;
    positive(&h)?;
    let r = rhs.as_ref();
    finite(r)?;
    if !bc.contains(&Boundary::Dirichlet) {
        let sum: f64 = r.iter().sum();
        let norm: f64 = r.iter().map(|a| a.abs()).sum();
        if sum.abs() > 1e-12 * norm {
            return Err(Error::Incompatible);
        }
    }
    let mut v: Vec<_> = r.iter().map(|&a| Z::from(a)).collect();
    let mut p = Plan::new();
    axis(&mut p, &mut v, g, Axis::X, bc[0], false);
    axis(&mut p, &mut v, g, Axis::Y, bc[1], false);
    divide(&mut v, g, h, bc);
    axis(&mut p, &mut v, g, Axis::Y, bc[1], true);
    axis(&mut p, &mut v, g, Axis::X, bc[0], true);
    real(v.into_iter().map(|a| a.re).collect())
}
/// Invert the periodic first derivative in Fourier space to obtain E from rho. Suppress the zero and even-grid Nyquist modes.
pub(super) fn field<A: Storage<f64>>(rho: &A, dx: f64) -> Result<A> {
    let n = rho.len();
    if n < 3 {
        return Err(Error::Argument);
    }
    positive(&[dx])?;
    let mut p = Plan::new();
    let mut z: Vec<_> = rho.as_ref().iter().map(|&a| Z::from(a)).collect();
    fft(&mut p, &mut z, false);
    // The constant field mode fixes the mean of E.
    z[0] = ZERO;
    for (k, a) in z.iter_mut().enumerate().skip(1) {
        let mode = if k <= n / 2 {
            k as f64
        } else {
            k as f64 - n as f64
        };
        *a = if 2 * k == n {
            ZERO
        } else {
            *a / Z::new(0., TAU * mode / (n as f64 * dx))
        };
    }
    fft(&mut p, &mut z, true);
    real(z.into_iter().map(|a| a.re).collect())
}
/// Build the tridiagonal y equations for one x sine mode, including y-dependent drift and reaction.
fn column(v: &[Z], g: Grid, k: usize, h: [f64; 2], b: &[f64], c: &[f64]) -> Vec<Equation> {
    let x = eigen(k, g.nx, h[0], Boundary::Dirichlet);
    let y = 1. / h[1].powi(2);
    (0..g.ny)
        .map(|j| {
            let w = b[j] / (2. * h[1]);
            Equation {
                l: if j == 0 { 0. } else { y - w },
                d: x - 2. * y + c[j],
                r: if j + 1 == g.ny { 0. } else { y + w },
                b: v[j * g.nx + k].re,
            }
        })
        .collect()
}
/// Transform along x, solve each mode's y equations by scalar cyclic reduction, then invert the transform.
pub(super) fn elliptic_cr<A: Storage<f64>>(
    rhs: &A,
    g: Grid,
    h: [f64; 2],
    b: &A,
    c: &A,
) -> Result<A> {
    g.check(rhs)?;
    positive(&h)?;
    if b.len() != g.ny || c.len() != g.ny {
        return Err(Error::Argument);
    }
    for x in [rhs, b, c] {
        finite(x.as_ref())?;
    }
    let mut v: Vec<_> = rhs.as_ref().iter().map(|&a| Z::from(a)).collect();
    let mut p = Plan::new();
    axis(&mut p, &mut v, g, Axis::X, Boundary::Dirichlet, false);
    for k in 0..g.nx {
        let sol = cr(&column(&v, g, k, h, b.as_ref(), c.as_ref()))?;
        for (j, a) in sol.into_iter().enumerate() {
            v[j * g.nx + k] = Z::from(a);
        }
    }
    axis(&mut p, &mut v, g, Axis::X, Boundary::Dirichlet, true);
    real(v.into_iter().map(|a| a.re).collect())
}
