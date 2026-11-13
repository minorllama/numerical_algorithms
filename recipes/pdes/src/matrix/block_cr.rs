//! Block cyclic reduction through tridiagonal factors of Chebyshev polynomials.
use super::{cpu::Storage, linear::solve, stencil::real, *};
use crate::{finite, positive};
use std::f64::consts::PI;

struct Block {
    lo: Vec<f64>,
    diag: Vec<f64>,
    hi: Vec<f64>,
}

impl Block {
    /// Build the original tridiagonal y block, scaled by dx² and including the −2 contribution from x diffusion.
    fn new(h: [f64; 2], drift: &[f64], reaction: &[f64]) -> Self {
        let r = (h[0] / h[1]).powi(2);
        let scale = h[0].powi(2) / (2. * h[1]);
        Self {
            lo: drift.iter().map(|b| r - scale * b).collect(),
            diag: reaction
                .iter()
                .map(|c| -2. - 2. * r + h[0].powi(2) * c)
                .collect(),
            hi: drift.iter().map(|b| r + scale * b).collect(),
        }
    }
}

/// Roots of the reduced diagonal block polynomial at this level.
///
/// These shifts factor the Chebyshev polynomial of the original block at the current level.
fn roots(degree: usize) -> impl Iterator<Item = f64> {
    (0..degree).map(move |k| 2. * (PI * (k as f64 + 0.5) / degree as f64).cos())
}

/// Multiply one y column by a shifted copy of the original tridiagonal block.
fn apply_shift(t: &Block, v: &[f64], z: f64) -> Vec<f64> {
    (0..v.len())
        .map(|j| {
            let l = j.checked_sub(1).map_or(0., |i| v[i]);
            let r = v.get(j + 1).copied().unwrap_or(0.);
            t.lo[j] * l + (t.diag[j] - z) * v[j] + t.hi[j] * r
        })
        .collect()
}

/// Apply a reduced block by successive shifted multiplications, with the sign of the reduced polynomial.
fn apply(t: &Block, mut v: Vec<f64>, degree: usize) -> Result<Vec<f64>> {
    for z in roots(degree) {
        v = apply_shift(t, &v, z);
    }
    if degree > 1 {
        v.iter_mut().for_each(|v| *v = -*v);
    }
    finite(&v)?;
    Ok(v)
}

/// Invert a reduced block by successive shifted Thomas solves, with the sign of the reduced polynomial.
fn invert(t: &Block, mut v: Vec<f64>, degree: usize) -> Result<Vec<f64>> {
    for z in roots(degree) {
        let d: Vec<_> = t.diag.iter().map(|a| a - z).collect();
        v = solve(&t.lo, &d, &t.hi, v)?;
    }
    if degree > 1 {
        v.iter_mut().for_each(|v| *v = -*v);
    }
    Ok(v)
}

/// Keep every other x column and form its new RHS from its two neighbors and the reduced diagonal block.
fn reduce(t: &Block, rows: &[Vec<f64>], degree: usize) -> Result<Vec<Vec<f64>>> {
    (1..rows.len())
        .step_by(2)
        .map(|j| {
            let a = apply(t, rows[j].clone(), degree)?;
            Ok(a.iter()
                .enumerate()
                .map(|(i, v)| rows[j - 1][i] + rows[j + 1][i] - v)
                .collect())
        })
        .collect()
}

/// Copy the retained columns back and solve for the eliminated columns using their saved equations.
fn restore(t: &Block, rows: &[Vec<f64>], x: Vec<Vec<f64>>, degree: usize) -> Result<Vec<Vec<f64>>> {
    let mut full = Vec::with_capacity(rows.len());
    for (j, row) in rows.iter().enumerate() {
        if j % 2 == 1 {
            full.push(x[j / 2].clone());
        } else {
            let rhs = row
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let l = (j / 2).checked_sub(1).map_or(0., |k| x[k][i]);
                    let r = x.get(j / 2).map_or(0., |v| v[i]);
                    v - l - r
                })
                .collect();
            full.push(invert(t, rhs, degree)?);
        }
    }
    Ok(full)
}

/// Reduce until a single x column remains, solve it and restore columns while unwinding.
fn recurse(t: &Block, rows: Vec<Vec<f64>>, degree: usize) -> Result<Vec<Vec<f64>>> {
    if rows.len() == 1 {
        return Ok(vec![invert(t, rows.into_iter().next().unwrap(), degree)?]);
    }
    // Elimination squares the block polynomial, doubling its degree.
    let x = recurse(t, reduce(t, &rows, degree)?, 2 * degree)?;
    restore(t, &rows, x, degree)
}

/// Validate the grid, gather x columns and solve by block reduction. Scatter the columns into a flat row-major array.
pub(super) fn block_cr<A: Storage<f64>>(rhs: &A, g: Grid, h: [f64; 2], b: &A, c: &A) -> Result<A> {
    g.check(rhs)?;
    positive(&h)?;
    if !g.nx.checked_add(1).is_some_and(usize::is_power_of_two)
        || b.len() != g.ny
        || c.len() != g.ny
    {
        return Err(Error::Argument);
    }
    for a in [rhs, b, c] {
        finite(a.as_ref())?;
    }
    let t = Block::new(h, b.as_ref(), c.as_ref());
    let rows = (0..g.nx)
        .map(|i| {
            (0..g.ny)
                .map(|j| h[0].powi(2) * rhs.as_ref()[j * g.nx + i])
                .collect()
        })
        .collect();
    let x = recurse(&t, rows, 1)?;
    real((0..rhs.len()).map(|i| x[i % g.nx][i / g.nx]).collect())
}
