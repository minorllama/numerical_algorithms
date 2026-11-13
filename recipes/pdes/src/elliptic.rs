//! Separable elliptic boundary-value problems (§20.4).
use crate::{matrix::Boundary, Backend, Grid, Result};

/// Solve the discrete Laplacian equation Δ_h u = rhs by FFT/DST/DCT.
///
/// Pass spacings and boundary types in x, y order. With no Dirichlet axis, the RHS must have zero mean and the solution is normalized to zero mean.
pub fn poisson<B: Backend>(
    b: &B,
    rhs: &B::Real,
    grid: Grid,
    h: [f64; 2],
    bc: [Boundary; 2],
) -> Result<B::Real> {
    b.poisson(rhs, grid, h, bc)
}

/// Solve Δ_h u + drift(y) D_y u + reaction(y) u = rhs.
/// Homogeneous Dirichlet data. A sine transform in x yields scalar cyclic
/// reduction systems in y. Requires nonzero elimination pivots.
///
/// Provide one drift and reaction coefficient per interior y row. The output uses the same shape and row-major layout as rhs.
pub fn cyclic_reduction<B: Backend>(
    b: &B,
    rhs: &B::Real,
    grid: Grid,
    h: [f64; 2],
    drift: &B::Real,
    reaction: &B::Real,
) -> Result<B::Real> {
    b.elliptic_cr(rhs, grid, h, drift, reaction)
}

/// Direct block cyclic reduction (§20.4.2), with 2^m-1 interior x nodes.
/// Homogeneous Dirichlet data and nonsingular tridiagonal factors.
///
/// Provide one drift and reaction coefficient per interior y row. This variant requires nx + 1 to be a power of two.
pub fn block_cyclic_reduction<B: Backend>(
    b: &B,
    rhs: &B::Real,
    grid: Grid,
    h: [f64; 2],
    drift: &B::Real,
    reaction: &B::Real,
) -> Result<B::Real> {
    b.block_cr(rhs, grid, h, drift, reaction)
}
