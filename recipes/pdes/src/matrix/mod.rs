//! Array storage and matrix operations. A device backend need not expose host slices.
mod block_cr;
mod cpu;
mod linear;
mod spectral;
mod stencil;
use crate::{Error, Result};
pub use cpu::Cpu;
pub use rustfft::num_complex::Complex64;

/// Storage contract for backend arrays. It makes no host accessibility assumption.
pub trait Array {
    type Item;
    /// Return the number of elements in storage order.
    fn len(&self) -> usize;
    /// Report whether the buffer has zero elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl<T> Array for Vec<T> {
    type Item = T;
    /// Return the number of elements in storage order.
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}
impl<T> Array for Box<[T]> {
    type Item = T;
    /// Return the number of elements in storage order.
    fn len(&self) -> usize {
        self.as_ref().len()
    }
}

/// Row-major grid, with x varying fastest.
#[derive(Debug, Clone, Copy)]
pub struct Grid {
    /// Number of unknowns along the contiguous x direction.
    pub nx: usize,
    /// Number of rows along y, or velocity rows for Vlasov.
    pub ny: usize,
}
impl Grid {
    /// Return the checked number of cells. Reject nx below three, ny of zero, or a product that overflows.
    pub fn size(self) -> Result<usize> {
        if self.nx < 3 || self.ny == 0 {
            return Err(Error::Argument);
        }
        self.nx.checked_mul(self.ny).ok_or(Error::Argument)
    }
    /// Verify that a buffer contains exactly one element per grid cell.
    pub fn check<A: Array>(self, data: &A) -> Result<()> {
        if self.size()? == data.len() {
            Ok(())
        } else {
            Err(Error::Argument)
        }
    }
}
#[derive(Clone, Copy)]
pub enum Axis {
    X,
    Y,
}

/// Tridiagonal coefficients, one entry per point, including endpoints.
#[derive(Clone)]
pub struct Tri<A> {
    /// Coefficient at row i, column i−1.
    pub lo: A,
    /// Coefficient at row i, column i.
    pub diag: A,
    /// Coefficient at row i, column i+1.
    pub hi: A,
}

/// Boundary on each axis of a separable Laplacian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    Periodic,
    /// Unknowns are the interior nodes, with zero values at the exterior nodes.
    Dirichlet,
    /// Cell-centered grid, with zero normal flux at its faces.
    Neumann,
}

#[derive(Debug, Clone, Copy)]
pub enum Flux {
    Linear(f64),
    Burgers,
}

#[derive(Debug, Clone, Copy)]
pub enum Scheme {
    Ftcs,
    Lax,
    LaxWendroff,
    Upwind,
    Godunov,
    Muscl,
}

/// Backend operations use abstract buffers and can dispatch to device kernels.
pub trait Backend {
    type Real: Array<Item = f64>;
    type Complex: Array<Item = Complex64>;

    /// Create a real backend buffer from host values.
    fn real(&self, values: &[f64]) -> Result<Self::Real>;
    /// Create a complex backend buffer from host values.
    fn complex(&self, values: &[Complex64]) -> Result<Self::Complex>;
    /// Return an owned host copy of a real backend buffer.
    fn to_host(&self, values: &Self::Real) -> Result<Vec<f64>>;
    /// Reduce a real buffer to its maximum absolute value for a CFL check.
    fn max_abs(&self, a: &Self::Real) -> Result<f64>;
    /// Advance one periodic scalar conservation step with ratio = dt / dx.
    fn flux(&self, u: &Self::Real, flux: Flux, scheme: Scheme, ratio: f64) -> Result<Self::Real>;
    /// Advance centered advection leapfrog from two time levels and a signed Courant number.
    fn leapfrog(&self, prev: &Self::Real, u: &Self::Real, c: f64) -> Result<Self::Real>;
    /// Advance the staggered wave pair by one step.
    fn wave(&self, p: &Self::Real, q: &Self::Real, c: f64) -> Result<(Self::Real, Self::Real)>;
    /// Apply the unsplit 2D Lax stencil with one Courant number per axis.
    fn lax2(&self, u: &Self::Real, grid: Grid, cx: f64, cy: f64) -> Result<Self::Real>;
    /// Advance independent lines in one direction with one speed per line and ratio = dt / spacing.
    fn sweep(
        &self,
        u: &Self::Real,
        grid: Grid,
        axis: Axis,
        speed: &Self::Real,
        ratio: f64,
    ) -> Result<Self::Real>;
    /// Integrate the phase-space buffer over its velocity rows.
    fn integrate_v(&self, u: &Self::Real, grid: Grid, dv: f64) -> Result<Self::Real>;
    /// Return the periodic electric field with zero mean from a spatial density.
    fn field(&self, density: &Self::Real, dx: f64) -> Result<Self::Real>;
    /// Advance variable face diffusivity with ratio = dt / dx² and the supplied theta weight.
    fn diffusion(
        &self,
        u: &Self::Real,
        faces: &Self::Real,
        ratio: f64,
        theta: f64,
    ) -> Result<Self::Real>;
    /// Advance u_t = z(u)_xx using z and its derivative d at the old state.
    fn nonlinear_diffusion(
        &self,
        u: &Self::Real,
        d: &Self::Real,
        z: &Self::Real,
        ratio: f64,
        implicit: bool,
    ) -> Result<Self::Real>;
    /// Advance a real tridiagonal generator with prescribed new endpoint values.
    fn theta(
        &self,
        u: &Self::Real,
        op: &Tri<Self::Real>,
        dt: f64,
        theta: f64,
        ends: [f64; 2],
    ) -> Result<Self::Real>;
    /// Advance a complex tridiagonal generator by Crank–Nicolson with zero endpoints.
    fn theta_complex(
        &self,
        u: &Self::Complex,
        op: &Tri<Self::Complex>,
        dt: f64,
    ) -> Result<Self::Complex>;
    /// Advance periodic constant diffusion by Crank–Nicolson with ratio = nu * dt / dx².
    fn periodic_diffusion(&self, u: &Self::Real, ratio: f64) -> Result<Self::Real>;
    /// Apply or invert one directional periodic stencil using the supplied dimensionless ratio.
    fn axis_diffusion(
        &self,
        u: &Self::Real,
        grid: Grid,
        axis: Axis,
        ratio: f64,
        implicit: bool,
    ) -> Result<Self::Real>;
    /// Solve the discrete Poisson equation with a boundary type and spacing for each axis.
    fn poisson(
        &self,
        rhs: &Self::Real,
        grid: Grid,
        h: [f64; 2],
        bc: [Boundary; 2],
    ) -> Result<Self::Real>;
    /// Solve a separable elliptic equation by a sine transform and scalar cyclic reduction.
    fn elliptic_cr(
        &self,
        rhs: &Self::Real,
        grid: Grid,
        h: [f64; 2],
        drift: &Self::Real,
        reaction: &Self::Real,
    ) -> Result<Self::Real>;
    /// Solve a separable elliptic equation by direct block cyclic reduction.
    fn block_cr(
        &self,
        rhs: &Self::Real,
        grid: Grid,
        h: [f64; 2],
        drift: &Self::Real,
        reaction: &Self::Real,
    ) -> Result<Self::Real>;
}
