//! Real and complex tridiagonal time stepping (§20.2).
use crate::Array;
use crate::{
    matrix::{Complex64, Tri},
    positive, Backend, Error, Result,
};

#[derive(Clone, Copy)]
pub enum TimeScheme {
    /// Forward Euler, subject to the explicit diffusion stability bound.
    Explicit,
    /// First-order implicit step with strong damping of stiff modes.
    BackwardEuler,
    /// Second-order trapezoidal step.
    CrankNicolson,
}
impl TimeScheme {
    /// Map the time integrator to the implicit weight in the theta method.
    fn weight(self) -> f64 {
        match self {
            Self::Explicit => 0.,
            Self::BackwardEuler => 1.,
            Self::CrankNicolson => 0.5,
        }
    }
}

/// Step u_t=(D(x) u_x)_x with fixed Dirichlet endpoints and n-1 face diffusivities.
///
/// Supply one face coefficient between each pair of adjacent nodes. The output contains the endpoints as well as the updated interior.
pub fn diffusion<B: Backend>(
    b: &B,
    u: &B::Real,
    faces: &B::Real,
    dt: f64,
    dx: f64,
    scheme: TimeScheme,
) -> Result<B::Real> {
    positive(&[dt, dx])?;
    b.diffusion(u, faces, dt / dx.powi(2), scheme.weight())
}

/// Step u_t=(z(u))_xx, where d=z'(u)>=0 is evaluated at the old state.
/// Uses forward Euler or a single linearization of backward Euler.
///
/// Supply one d and z value per grid point and ratio = dt / dx². Recompute d and z from the current state before calling again.
pub fn nonlinear<B: Backend>(
    b: &B,
    u: &B::Real,
    d: &B::Real,
    z: &B::Real,
    ratio: f64,
    implicit: bool,
) -> Result<B::Real> {
    positive(&[ratio])?;
    b.nonlinear_diffusion(u, d, z, ratio, implicit)
}

/// Advance i psi_t = -psi_xx + V psi with zero Dirichlet endpoints.
/// The complex generator should contain i/dx² off-diagonals and
/// -i(2/dx²+V_i) on the diagonal. Initial endpoints must be zero.
///
/// Build the generator once with schrodinger_operator for a static potential. Feed the returned complex array into the next time step.
pub fn schrodinger<B: Backend>(
    b: &B,
    u: &B::Complex,
    generator: &Tri<B::Complex>,
    dt: f64,
) -> Result<B::Complex> {
    positive(&[dt])?;
    b.theta_complex(u, generator, dt)
}

/// Assemble the generator for the preceding Schrödinger step.
///
/// Supply one potential value per node, including the two endpoints.
pub fn schrodinger_operator<B: Backend>(
    b: &B,
    potential: &[f64],
    dx: f64,
) -> Result<Tri<B::Complex>> {
    positive(&[dx])?;
    crate::finite(potential)?;
    if potential.len() < 3 {
        return Err(Error::Argument);
    }
    let a = 1. / dx.powi(2);
    let side = vec![Complex64::new(0., a); potential.len()];
    let diag: Vec<_> = potential
        .iter()
        .map(|v| Complex64::new(0., -2. * a - v))
        .collect();
    Ok(Tri {
        lo: b.complex(&side)?,
        diag: b.complex(&diag)?,
        hi: b.complex(&side)?,
    })
}

/// Generic theta step for a linear, time-independent tridiagonal generator.
///
/// Supply generator coefficients with dx factors already included. The old endpoint values are in u and ends sets the endpoints at the new time.
pub fn theta<B: Backend>(
    b: &B,
    u: &B::Real,
    generator: &Tri<B::Real>,
    dt: f64,
    scheme: TimeScheme,
    ends: [f64; 2],
) -> Result<B::Real> {
    positive(&[dt])?;
    crate::finite(&ends)?;
    b.theta(u, generator, dt, scheme.weight(), ends)
}

#[derive(Clone, Copy)]
pub struct Call {
    /// Spot price at the rightmost grid node.
    pub s_max: f64,
    /// Strike used in the payoff and the right endpoint condition.
    pub strike: f64,
    /// Constant continuously compounded risk-free rate.
    pub rate: f64,
    /// Positive constant volatility.
    pub volatility: f64,
}
/// Assemble the centered Black–Scholes generator. On the uniform spot grid, S_j / dS = j, so dS cancels from these coefficients.
fn call_operator<B: Backend>(b: &B, n: usize, p: Call) -> Result<Tri<B::Real>> {
    let s: Vec<_> = (0..n).map(|j| j as f64).collect();
    let q = |j: f64| 0.5 * (p.volatility * j).powi(2);
    let lo: Vec<_> = s.iter().map(|&j| q(j) - 0.5 * p.rate * j).collect();
    let hi: Vec<_> = s.iter().map(|&j| q(j) + 0.5 * p.rate * j).collect();
    let diag: Vec<_> = s.iter().map(|&j| -2. * q(j) - p.rate).collect();
    Ok(Tri {
        lo: b.real(&lo)?,
        diag: b.real(&diag)?,
        hi: b.real(&hi)?,
    })
}
/// Return the left endpoint and the far-field approximation at time to maturity t.
fn call_boundary(p: Call, t: f64) -> [f64; 2] {
    [0., p.s_max - p.strike * (-p.rate * t).exp()]
}

/// Price a European call using Crank–Nicolson with two initial implicit half-steps.
/// u holds the payoff on a uniform S grid, including S=0 and S=s_max.
///
/// Consumes the initial array and returns the price at time to maturity dt * steps. Sample max(S − strike, 0) at all nodes, including both ends.
pub fn black_scholes<B: Backend>(
    b: &B,
    u: B::Real,
    p: Call,
    dt: f64,
    steps: usize,
) -> Result<B::Real> {
    positive(&[dt, p.s_max, p.strike, p.volatility])?;
    crate::finite(&[p.rate])?;
    if u.len() < 3 || steps == 0 || p.s_max <= p.strike {
        return Err(Error::Argument);
    }
    let a = call_operator(b, u.len(), p)?;
    // Smooth the payoff kink over the first full interval.
    let half = b.theta(&u, &a, 0.5 * dt, 1., call_boundary(p, 0.5 * dt))?;
    let mut u = b.theta(&half, &a, 0.5 * dt, 1., call_boundary(p, dt))?;
    for j in 1..steps {
        u = b.theta(&u, &a, dt, 0.5, call_boundary(p, (j + 1) as f64 * dt))?;
    }
    Ok(u)
}
