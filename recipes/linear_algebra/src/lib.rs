//! Classical real linear algebra, implemented independently in safe Rust.
//! See README.md for conventions, algorithms, complexity and limitations.
#![forbid(unsafe_code)]

pub mod fast;
use std::ops::{Index, IndexMut};


#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Dimension,
    NonFinite,
    InvalidParameter,
    Singular,
    NotSymmetric,
    NotPositiveDefinite,
    Breakdown,
    NoConvergence { iterations: usize, residual: f64 },
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// Nonempty row-major matrix; immutable dimensions and owned entries.
/// Constructors reject nonfinite entries; indexing permits subsequent mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(data.len()) {
            return Err(Error::Dimension);
        }
        finite(&data)?;
        Ok(Self { rows, cols, data })
    }
    pub fn zeros(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "matrix dimensions must be nonzero");
        Self {
            rows,
            cols,
            data: vec![0.; rows.checked_mul(cols).expect("matrix too large")],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut a = Self::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 1.;
        }
        a
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    pub fn transpose(&self) -> Self {
        let mut a = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                a[(j, i)] = self[(i, j)];
            }
        }
        a
    }
    pub fn mul(&self, b: &Self) -> Result<Self> {
        if self.cols != b.rows {
            return Err(Error::Dimension);
        }
        let mut c = Self::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                for j in 0..b.cols {
                    c[(i, j)] += self[(i, k)] * b[(k, j)];
                }
            }
        }
        finite(&c.data)?;
        Ok(c)
    }
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>> {
        vector(x, self.cols)?;
        let y = (0..self.rows)
            .map(|i| dot(&self.data[i * self.cols..(i + 1) * self.cols], x))
            .collect::<Vec<_>>();
        finite(&y)?;
        Ok(y)
    }
    pub fn norm_inf(&self) -> f64 {
        self.data
            .chunks(self.cols)
            .map(|r| r.iter().map(|x| x.abs()).sum::<f64>())
            .fold(0., f64::max)
    }
    pub(crate) fn square(&self) -> Result<usize> {
        finite(&self.data)?;
        if self.rows != self.cols {
            Err(Error::Dimension)
        } else {
            Ok(self.rows)
        }
    }
    pub(crate) fn swap_rows(&mut self, i: usize, j: usize) {
        for k in 0..self.cols {
            self.data.swap(i * self.cols + k, j * self.cols + k);
        }
    }
}
impl Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        assert!(i < self.rows && j < self.cols);
        &self.data[i * self.cols + j]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        assert!(i < self.rows && j < self.cols);
        &mut self.data[i * self.cols + j]
    }
}
pub(crate) fn finite(x: &[f64]) -> Result<()> {
    if x.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite)
    }
}
pub(crate) fn vector(x: &[f64], n: usize) -> Result<()> {
    if x.len() != n {
        return Err(Error::Dimension);
    }
    finite(x)
}
pub(crate) fn tolerance(t: f64) -> Result<()> {
    if t.is_finite() && t >= 0. {
        Ok(())
    } else {
        Err(Error::InvalidParameter)
    }
}
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// Overflow-resistant Euclidean norm (same primitive as the two-argument hypot).
pub fn norm2(x: &[f64]) -> f64 {
    x.iter().fold(0., |a, &b| a.hypot(b))
}
/// Relative infinity-norm backward error, ||b-Ax||/(||A|| ||x||+||b||).
pub fn backward_error(a: &Matrix, x: &[f64], b: &[f64]) -> Result<f64> {
    vector(b, a.rows)?;
    let ax = a.mul_vec(x)?;
    let r = ax
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max);
    let den = a.norm_inf() * x.iter().map(|v| v.abs()).fold(0., f64::max)
        + b.iter().map(|v| v.abs()).fold(0., f64::max);
    Ok(if den == 0. { r } else { r / den })
}
