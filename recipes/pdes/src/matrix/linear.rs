use super::{
    cpu::Storage,
    stencil::{cell, dimensions, left, real, right},
    *,
};
use crate::{cfl, finite};
use std::ops::{Add, Div, Mul, Sub};

pub(super) trait Scalar:
    Copy + From<f64> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    /// Return the scalar magnitude used to check coefficients and pivots.
    fn magnitude(self) -> f64;
}
impl Scalar for f64 {
    /// Return the scalar magnitude used to check coefficients and pivots.
    fn magnitude(self) -> f64 {
        self.abs()
    }
}
impl Scalar for Complex64 {
    /// Return the scalar magnitude used to check coefficients and pivots.
    fn magnitude(self) -> f64 {
        self.norm()
    }
}
/// Reject a real or complex buffer with nonfinite magnitude.
fn check<T: Scalar>(v: &[T]) -> Result<()> {
    if v.iter().all(|x| x.magnitude().is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
/// Reject a zero or nonfinite elimination pivot.
fn pivot<T: Scalar>(p: T) -> Result<()> {
    let x = p.magnitude();
    if !x.is_finite() {
        Err(Error::NonFinite)
    } else if x == 0. {
        Err(Error::Singular)
    } else {
        Ok(())
    }
}

/// Thomas elimination for real or complex coefficients.
///
/// Overwrite the working diagonal and right-hand side in a forward sweep, then back-substitute. Off-diagonal entries are read-only.
pub(super) fn solve<T: Scalar>(a: &[T], d: &[T], c: &[T], mut b: Vec<T>) -> Result<Vec<T>> {
    let n = b.len();
    if n == 0 || [a.len(), d.len(), c.len()] != [n; 3] {
        return Err(Error::Argument);
    }
    let mut d = d.to_vec();
    for i in 1..n {
        pivot(d[i - 1])?;
        let t = a[i] / d[i - 1];
        d[i] = d[i] - t * c[i - 1];
        b[i] = b[i] - t * b[i - 1];
    }
    for i in (0..n).rev() {
        pivot(d[i])?;
        if i + 1 < n {
            b[i] = b[i] - c[i] * b[i + 1];
        }
        b[i] = b[i] / d[i];
    }
    check(&b)?;
    Ok(b)
}

/// Apply the explicit part of the theta update to the old state and set the new boundary values on the RHS.
fn theta_rhs<T: Scalar>(u: &[T], op: [&[T]; 3], s: T, ends: [T; 2]) -> Vec<T> {
    let n = u.len();
    let [l, d, h] = op;
    let mut b = u.to_vec();
    b[0] = ends[0];
    b[n - 1] = ends[1];
    for i in 1..n - 1 {
        b[i] = u[i] + s * (l[i] * u[i - 1] + d[i] * u[i] + h[i] * u[i + 1]);
    }
    b
}

/// Form the implicit tridiagonal matrix with identity rows at the two endpoints.
fn theta_matrix<T: Scalar>(n: usize, op: [&[T]; 3], s: T) -> [Vec<T>; 3] {
    let z = T::from(0.);
    let o = T::from(1.);
    let mut a = [vec![z; n], vec![o; n], vec![z; n]];
    for i in 1..n - 1 {
        a[0][i] = z - s * op[0][i];
        a[1][i] = o - s * op[1][i];
        a[2][i] = z - s * op[2][i];
    }
    a
}

/// Apply one generic real or complex theta step. At theta = 0 the RHS is already the solution.
pub(super) fn theta<T: Scalar, A: Storage<T>>(
    u: &A,
    op: &Tri<A>,
    dt: f64,
    w: f64,
    ends: [T; 2],
) -> Result<A> {
    let n = u.len();
    if n < 3 || [op.lo.len(), op.diag.len(), op.hi.len()] != [n; 3] {
        return Err(Error::Argument);
    }
    let coeff = [op.lo.as_ref(), op.diag.as_ref(), op.hi.as_ref()];
    check(u.as_ref())?;
    let b = theta_rhs(u.as_ref(), coeff, T::from((1. - w) * dt), ends);
    if w == 0. {
        check(&b)?;
        return Ok(b.into());
    }
    let [a, d, c] = theta_matrix(n, coeff, T::from(w * dt));
    Ok(solve(&a, &d, &c, b)?.into())
}

/// Build the conservative variable-diffusion stencil from the left and right face coefficients at each node.
fn face_operator<A: Storage<f64>>(f: &[f64]) -> Tri<A> {
    let n = f.len() + 1;
    let mut a = vec![0.; n];
    let mut b = vec![0.; n];
    let mut c = vec![0.; n];
    for i in 1..n - 1 {
        a[i] = f[i - 1];
        c[i] = f[i];
        b[i] = -a[i] - c[i];
    }
    Tri {
        lo: a.into(),
        diag: b.into(),
        hi: c.into(),
    }
}

/// Keep the endpoints fixed and dispatch the face-based diffusion problem to the theta method.
pub(super) fn diffusion<A: Storage<f64>>(u: &A, faces: &A, ratio: f64, w: f64) -> Result<A> {
    if u.len() < 3 || faces.len() != u.len() - 1 {
        return Err(Error::Argument);
    }
    let f = faces.as_ref();
    finite(f)?;
    if f.iter().any(|&x| x < 0.) {
        return Err(Error::Argument);
    }
    if w == 0. {
        cfl(2. * ratio * f.iter().copied().fold(0., f64::max))?;
    }
    let u0 = u.as_ref();
    theta(u, &face_operator(f), ratio, w, [u0[0], u0[u.len() - 1]])
}

/// Form the second difference of z(u). Use it directly for an explicit increment, or solve for a linearized implicit increment.
pub(super) fn nonlinear_diffusion<A: Storage<f64>>(
    u: &A,
    d: &A,
    z: &A,
    r: f64,
    implicit: bool,
) -> Result<A> {
    let n = u.len();
    if n < 3 || d.len() != n || z.len() != n {
        return Err(Error::Argument);
    }
    let (u, d, z) = (u.as_ref(), d.as_ref(), z.as_ref());
    finite(d)?;
    if d.iter().any(|&x| x < 0.) {
        return Err(Error::Argument);
    }
    let mut delta = vec![0.; n];
    for i in 1..n - 1 {
        delta[i] = r * (z[i - 1] - 2. * z[i] + z[i + 1]);
    }
    if implicit {
        delta = solve_nonlinear_delta(d, r, delta)?;
    } else {
        cfl(2. * r * d.iter().copied().fold(0., f64::max))?;
    }
    real(u.iter().zip(delta).map(|(u, du)| u + du).collect())
}

/// Solve for the nonlinear diffusion increment with zero endpoint increments. Each off-diagonal uses z-prime at its neighboring node.
fn solve_nonlinear_delta(d: &[f64], r: f64, delta: Vec<f64>) -> Result<Vec<f64>> {
    let n = d.len();
    let (mut a, mut b, mut c) = (vec![0.; n], vec![1.; n], vec![0.; n]);
    for i in 1..n - 1 {
        a[i] = -r * d[i - 1];
        b[i] += 2. * r * d[i];
        c[i] = -r * d[i + 1];
    }
    solve(&a, &b, &c, delta)
}

/// Symmetric periodic (I-r Dxx) solve using a rank-one correction.
///
/// Use two tridiagonal solves and a Sherman–Morrison correction to restore the periodic corner couplings.
fn cyclic(u: Vec<f64>, r: f64) -> Result<Vec<f64>> {
    let n = u.len();
    let a = vec![-r; n];
    let d0 = 1. + 2. * r;
    // Split the periodic matrix into a tridiagonal part and one outer product.
    let gamma = -d0;
    let mut d = vec![d0; n];
    d[0] -= gamma;
    d[n - 1] -= r * r / gamma;
    let mut update = vec![0.; n];
    update[0] = gamma;
    update[n - 1] = -r;
    let x = solve(&a, &d, &a, u)?;
    let z = solve(&a, &d, &a, update)?;
    let beta = -r / gamma;
    let t = (x[0] + beta * x[n - 1]) / (1. + z[0] + beta * z[n - 1]);
    Ok(x.into_iter().zip(z).map(|(x, z)| x - t * z).collect())
}

/// Apply I + r L, where L is the periodic second difference without the spacing factor.
fn explicit_periodic(u: &[f64], r: f64) -> Vec<f64> {
    let n = u.len();
    (0..n)
        .map(|i| u[i] + r * (u[left(i, n)] - 2. * u[i] + u[right(i, n)]))
        .collect()
}
/// Compose the explicit half of the Crank–Nicolson update with the implicit periodic half.
pub(super) fn periodic_cn<A: Storage<f64>>(u: &A, r: f64) -> Result<A> {
    if u.len() < 3 || r < 0. || !r.is_finite() {
        return Err(Error::Argument);
    }
    real(cyclic(explicit_periodic(u.as_ref(), 0.5 * r), 0.5 * r)?)
}
/// Gather one line at a time, apply its explicit stencil or implicit solve, then scatter it back to the 2D output.
pub(super) fn axis_diffusion<A: Storage<f64>>(
    u: &A,
    g: Grid,
    axis: Axis,
    r: f64,
    implicit: bool,
) -> Result<A> {
    g.check(u)?;
    let (n, lanes) = dimensions(g, axis);
    if n < 3 {
        return Err(Error::Argument);
    }
    let mut y = vec![0.; u.len()];
    for j in 0..lanes {
        let line: Vec<_> = (0..n).map(|i| u.as_ref()[cell(g, axis, j, i)]).collect();
        let row = if implicit {
            cyclic(line, r)?
        } else {
            explicit_periodic(&line, r)
        };
        for (i, z) in row.into_iter().enumerate() {
            y[cell(g, axis, j, i)] = z;
        }
    }
    real(y)
}

#[derive(Clone, Copy)]
pub(super) struct Equation {
    pub l: f64,
    pub d: f64,
    pub r: f64,
    pub b: f64,
}

/// Substitute the adjacent odd-row equations into a retained even row, yielding couplings to its next retained neighbors.
fn eliminate(
    mut row: Equation,
    left: Option<Equation>,
    right: Option<Equation>,
) -> Result<Equation> {
    if let Some(x) = left {
        pivot(x.d)?;
        let q = row.l / x.d;
        row.l = -q * x.l;
        row.d -= q * x.r;
        row.b -= q * x.b;
    }
    if let Some(x) = right {
        pivot(x.d)?;
        let q = row.r / x.d;
        row.r = -q * x.r;
        row.d -= q * x.l;
        row.b -= q * x.b;
    }
    Ok(row)
}

/// Eliminate odd rows recursively, then recover them in reverse order.
///
/// Keep alternate rows until the reduced system has one unknown, then reconstruct the eliminated rows using the saved equations.
pub(super) fn cr(rows: &[Equation]) -> Result<Vec<f64>> {
    if rows.len() == 1 {
        pivot(rows[0].d)?;
        return Ok(vec![rows[0].b / rows[0].d]);
    }
    let even: Vec<_> = (0..rows.len())
        .step_by(2)
        .map(|i| {
            eliminate(
                rows[i],
                i.checked_sub(1).map(|j| rows[j]),
                rows.get(i + 1).copied(),
            )
        })
        .collect::<Result<_>>()?;
    let mut x = vec![0.; rows.len()];
    for (i, y) in cr(&even)?.into_iter().enumerate() {
        x[2 * i] = y;
    }
    for i in (1..rows.len()).step_by(2) {
        let row = rows[i];
        pivot(row.d)?;
        x[i] = (row.b - row.l * x[i - 1] - row.r * x.get(i + 1).copied().unwrap_or(0.)) / row.d;
    }
    finite(&x)?;
    Ok(x)
}
