use super::{linear, spectral, stencil, *};
use crate::finite;
use std::marker::PhantomData;

/// A CPU backend with independently selectable real and complex storage.
pub struct Cpu<R = Vec<f64>, C = Vec<Complex64>>(PhantomData<(R, C)>);
impl<R, C> Default for Cpu<R, C> {
    /// Construct the stateless CPU backend for the chosen real and complex storage types.
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub(super) trait Storage<T>: Array<Item = T> + AsRef<[T]> + From<Vec<T>> {}
impl<T, A> Storage<T> for A where A: Array<Item = T> + AsRef<[T]> + From<Vec<T>> {}

impl<R: Storage<f64>, C: Storage<Complex64>> Backend for Cpu<R, C> {
    type Real = R;
    type Complex = C;
    /// Create a real backend buffer from host values.
    fn real(&self, values: &[f64]) -> Result<R> {
        finite(values)?;
        Ok(values.to_vec().into())
    }
    /// Create a complex backend buffer from host values.
    fn complex(&self, values: &[Complex64]) -> Result<C> {
        if values
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(Error::NonFinite);
        }
        Ok(values.to_vec().into())
    }
    /// Return an owned host copy of a real backend buffer.
    fn to_host(&self, values: &R) -> Result<Vec<f64>> {
        finite(values.as_ref())?;
        Ok(values.as_ref().to_vec())
    }
    /// Reduce a real buffer to its maximum absolute value for a CFL check.
    fn max_abs(&self, a: &R) -> Result<f64> {
        finite(a.as_ref())?;
        Ok(a.as_ref().iter().map(|v| v.abs()).fold(0., f64::max))
    }
    /// Advance one periodic scalar conservation step with ratio = dt / dx.
    fn flux(&self, u: &R, f: Flux, s: Scheme, ratio: f64) -> Result<R> {
        stencil::flux(u, f, s, ratio)
    }
    /// Advance centered advection leapfrog from two time levels and a signed Courant number.
    fn leapfrog(&self, prev: &R, u: &R, c: f64) -> Result<R> {
        stencil::leapfrog(prev, u, c)
    }
    /// Advance the staggered wave pair by one step.
    fn wave(&self, p: &R, q: &R, c: f64) -> Result<(R, R)> {
        stencil::wave(p, q, c)
    }
    /// Apply the unsplit 2D Lax stencil with one Courant number per axis.
    fn lax2(&self, u: &R, g: Grid, cx: f64, cy: f64) -> Result<R> {
        stencil::lax2(u, g, cx, cy)
    }
    /// Advance independent lines in one direction with one speed per line and ratio = dt / spacing.
    fn sweep(&self, u: &R, g: Grid, axis: Axis, speed: &R, ratio: f64) -> Result<R> {
        stencil::sweep(u, g, axis, speed, ratio)
    }
    /// Integrate the phase-space buffer over its velocity rows.
    fn integrate_v(&self, u: &R, g: Grid, dv: f64) -> Result<R> {
        stencil::integrate_v(u, g, dv)
    }
    /// Return the periodic electric field with zero mean from a spatial density.
    fn field(&self, density: &R, dx: f64) -> Result<R> {
        spectral::field(density, dx)
    }
    /// Advance variable face diffusivity with ratio = dt / dx² and the supplied theta weight.
    fn diffusion(&self, u: &R, faces: &R, ratio: f64, theta: f64) -> Result<R> {
        linear::diffusion(u, faces, ratio, theta)
    }
    /// Advance u_t = z(u)_xx using z and its derivative d at the old state.
    fn nonlinear_diffusion(&self, u: &R, d: &R, z: &R, ratio: f64, implicit: bool) -> Result<R> {
        linear::nonlinear_diffusion(u, d, z, ratio, implicit)
    }
    /// Advance a real tridiagonal generator with prescribed new endpoint values.
    fn theta(&self, u: &R, op: &Tri<R>, dt: f64, theta: f64, ends: [f64; 2]) -> Result<R> {
        linear::theta(u, op, dt, theta, ends)
    }
    /// Advance a complex tridiagonal generator by Crank–Nicolson with zero endpoints.
    fn theta_complex(&self, u: &C, op: &Tri<C>, dt: f64) -> Result<C> {
        linear::theta(u, op, dt, 0.5, [Complex64::default(); 2])
    }
    /// Advance periodic constant diffusion by Crank–Nicolson with ratio = nu * dt / dx².
    fn periodic_diffusion(&self, u: &R, ratio: f64) -> Result<R> {
        linear::periodic_cn(u, ratio)
    }
    /// Apply or invert one directional periodic stencil using the supplied dimensionless ratio.
    fn axis_diffusion(&self, u: &R, g: Grid, axis: Axis, ratio: f64, implicit: bool) -> Result<R> {
        linear::axis_diffusion(u, g, axis, ratio, implicit)
    }
    /// Solve the discrete Poisson equation with a boundary type and spacing for each axis.
    fn poisson(&self, rhs: &R, grid: Grid, h: [f64; 2], bc: [Boundary; 2]) -> Result<R> {
        spectral::poisson(rhs, grid, h, bc)
    }
    /// Solve a separable elliptic equation by a sine transform and scalar cyclic reduction.
    fn elliptic_cr(&self, rhs: &R, grid: Grid, h: [f64; 2], drift: &R, reaction: &R) -> Result<R> {
        spectral::elliptic_cr(rhs, grid, h, drift, reaction)
    }
    /// Solve a separable elliptic equation by direct block cyclic reduction.
    fn block_cr(&self, rhs: &R, g: Grid, h: [f64; 2], drift: &R, reaction: &R) -> Result<R> {
        super::block_cr::block_cr(rhs, g, h, drift, reaction)
    }
}
