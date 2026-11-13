//! CPU finite difference solvers with a backend-independent Rust interface.
//!
//! Create a [`Cpu`] and pass it to the solver for your equation. Start with
//! [`multidim::burgers`], [`multidim::vlasov`], [`diffusion::black_scholes`],
//! or [`elliptic::poisson`]. Use the one-step functions for a custom time loop.
//! See the packaged RUST_API.md and the solve_pdes example for grid setup and
//! complete calls. The C ABI in [`ffi`] is for external callers.
pub mod diffusion;
pub mod elliptic;
pub mod ffi;
pub mod flux;
pub mod matrix;
pub mod multidim;

pub use matrix::{Array, Backend, Cpu, Grid};
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Error {
    Argument = 1,
    NonFinite = 2,
    Singular = 3,
    Cfl = 4,
    Incompatible = 5,
    Backend = 6,
}
impl std::fmt::Display for Error {
    /// Print the symbolic error name for Rust callers.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// Reject nonpositive or nonfinite spacings and time steps.
pub(crate) fn positive(xs: &[f64]) -> Result<()> {
    if xs.iter().all(|x| x.is_finite() && *x > 0.) {
        Ok(())
    } else {
        Err(Error::Argument)
    }
}
/// Reject input or output containing NaN or infinity.
pub(crate) fn finite(xs: &[f64]) -> Result<()> {
    if xs.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
/// Enforce a normalized CFL bound of one with a small roundoff allowance.
pub(crate) fn cfl(c: f64) -> Result<()> {
    if c.is_finite() && c <= 1. + 16. * f64::EPSILON {
        Ok(())
    } else {
        Err(Error::Cfl)
    }
}
