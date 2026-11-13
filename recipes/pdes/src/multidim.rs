//! Unsplit transport, ADI diffusion and Strang splitting (§20.3).
use crate::{
    cfl, flux,
    matrix::{Axis, Flux, Scheme},
    positive, Backend, Error, Grid, Result,
};

/// Advance periodic 2D constant-speed transport by one unsplit Lax step. Pass c = [a_x * dt / dx, a_y * dt / dy].
pub fn lax2<B: Backend>(b: &B, u: &B::Real, grid: Grid, c: [f64; 2]) -> Result<B::Real> {
    cfl(2. * (c[0] * c[0] + c[1] * c[1]))?;
    b.lax2(u, grid, c[0], c[1])
}

/// Peaceman–Rachford ADI for periodic u_t = D_x u_xx + D_y u_yy.
/// ratios are [dt D_x / dx², dt D_y / dy²].
///
/// The input and output are flat row-major arrays with x varying fastest. Call once per timestep.
pub fn adi_step<B: Backend>(b: &B, u: &B::Real, grid: Grid, r: [f64; 2]) -> Result<B::Real> {
    if r.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err(Error::Argument);
    }
    // First half-step is explicit in y and implicit in x.
    let y = b.axis_diffusion(u, grid, Axis::Y, 0.5 * r[1], false)?;
    let x = b.axis_diffusion(&y, grid, Axis::X, 0.5 * r[0], true)?;
    // Swap the explicit and implicit directions for the second half-step.
    let x = b.axis_diffusion(&x, grid, Axis::X, 0.5 * r[0], false)?;
    b.axis_diffusion(&x, grid, Axis::Y, 0.5 * r[1], true)
}

/// Compose A(dt/2), B(dt), A(dt/2).
///
/// Each closure takes ownership of its input state and returns the advanced state. The A closure is called twice per step.
pub fn strang<A, F, G>(u: A, dt: f64, mut a: F, mut b: G) -> Result<A>
where
    F: FnMut(A, f64) -> Result<A>,
    G: FnMut(A, f64) -> Result<A>,
{
    let x = a(u, 0.5 * dt)?;
    let x = b(x, dt)?;
    a(x, 0.5 * dt)
}

/// Periodic viscous Burgers with Lax–Wendroff or a limited advective flux.
///
/// Sample the initial condition on a periodic grid without the duplicate endpoint. Returns the profile after dt * steps.
pub fn burgers<B: Backend>(
    b: &B,
    mut u: B::Real,
    nu: f64,
    dx: f64,
    dt: f64,
    steps: usize,
    scheme: Scheme,
) -> Result<B::Real> {
    positive(&[nu, dx, dt])?;
    if steps == 0
        || !matches!(
            scheme,
            Scheme::LaxWendroff | Scheme::Muscl | Scheme::Godunov
        )
    {
        return Err(Error::Argument);
    }
    for _ in 0..steps {
        u = strang(
            u,
            dt,
            |x, t| b.periodic_diffusion(&x, nu * t / dx.powi(2)),
            |x, t| flux::step(b, &x, Flux::Burgers, scheme, t, dx),
        )?;
    }
    Ok(u)
}

#[derive(Clone, Copy)]
pub struct PhaseSpace {
    /// Spatial point count nx and velocity point count ny.
    pub grid: Grid,
    /// Positive spatial grid spacing.
    pub dx: f64,
    /// Positive velocity grid spacing.
    pub dv: f64,
    /// Velocity at the first row.
    pub v_min: f64,
}
impl PhaseSpace {
    /// Build one velocity per phase-space row, from v_min in increments of dv.
    fn speeds<B: Backend>(self, b: &B) -> Result<B::Real> {
        positive(&[self.dx, self.dv])?;
        crate::finite(&[self.v_min])?;
        let v: Vec<_> = (0..self.grid.ny)
            .map(|j| self.v_min + j as f64 * self.dv)
            .collect();
        b.real(&v)
    }
}

/// Vlasov–Poisson in 1x1v, with E_x=rho-mean(rho), mean(E)=0.
/// State is periodic in x and truncated/periodic in v.
///
/// Supply phase-space rows indexed by velocity with x varying fastest. Returns the state at dt * steps. Choose velocity bounds with negligible tails.
pub fn vlasov<B: Backend>(
    b: &B,
    mut u: B::Real,
    p: PhaseSpace,
    dt: f64,
    steps: usize,
) -> Result<B::Real> {
    p.grid.check(&u)?;
    positive(&[dt])?;
    if steps == 0 || p.grid.ny < 3 {
        return Err(Error::Argument);
    }
    let v = p.speeds(b)?;
    for _ in 0..steps {
        u = strang(
            u,
            dt,
            |x, t| b.sweep(&x, p.grid, Axis::X, &v, t / p.dx),
            |x, t| kick(b, &x, p, t),
        )?;
    }
    Ok(u)
}
/// Integrate the current distribution over velocity, solve for the electric field, then advect each spatial column in velocity.
fn kick<B: Backend>(b: &B, x: &B::Real, p: PhaseSpace, dt: f64) -> Result<B::Real> {
    let density = b.integrate_v(x, p.grid, p.dv)?;
    let e = b.field(&density, p.dx)?;
    b.sweep(x, p.grid, Axis::Y, &e, dt / p.dv)
}
