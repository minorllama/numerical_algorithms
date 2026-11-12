//! Strassen multiplication: the subcubic construction discussed in section 2.11.
use crate::*;
/// Square multiplication with zero padding to a power of two. `leaf` >= 1.
/// Extra allocation is intentional for clarity; conventional multiplication
/// is usually faster for small matrices.
pub fn strassen(a: &Matrix, b: &Matrix, leaf: usize) -> Result<Matrix> {
    let n = a.square()?;
    if b.square()? != n {
        return Err(Error::Dimension);
    }
    if leaf == 0 {
        return Err(Error::InvalidParameter);
    }
    let p = n.checked_next_power_of_two().ok_or(Error::Dimension)?;
    let mut x = Matrix::zeros(p, p);
    let mut y = x.clone();
    for i in 0..n {
        for j in 0..n {
            x[(i, j)] = a[(i, j)];
            y[(i, j)] = b[(i, j)];
        }
    }
    let z = rec(&x, &y, leaf)?;
    let mut out = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            out[(i, j)] = z[(i, j)];
        }
    }
    Ok(out)
}
fn add(a: &Matrix, b: &Matrix, sign: f64) -> Matrix {
    let mut c = a.clone();
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            c[(i, j)] += sign * b[(i, j)];
        }
    }
    c
}
fn block(a: &Matrix, i: usize, j: usize, n: usize) -> Matrix {
    let mut b = Matrix::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            b[(r, c)] = a[(i + r, j + c)];
        }
    }
    b
}
fn rec(a: &Matrix, b: &Matrix, leaf: usize) -> Result<Matrix> {
    let n = a.rows();
    if n <= leaf {
        return a.mul(b);
    }
    let h = n / 2;
    let (a11, a12, a21, a22) = (
        block(a, 0, 0, h),
        block(a, 0, h, h),
        block(a, h, 0, h),
        block(a, h, h, h),
    );
    let (b11, b12, b21, b22) = (
        block(b, 0, 0, h),
        block(b, 0, h, h),
        block(b, h, 0, h),
        block(b, h, h, h),
    );
    let m1 = rec(&add(&a11, &a22, 1.), &add(&b11, &b22, 1.), leaf)?;
    let m2 = rec(&add(&a21, &a22, 1.), &b11, leaf)?;
    let m3 = rec(&a11, &add(&b12, &b22, -1.), leaf)?;
    let m4 = rec(&a22, &add(&b21, &b11, -1.), leaf)?;
    let m5 = rec(&add(&a11, &a12, 1.), &b22, leaf)?;
    let m6 = rec(&add(&a21, &a11, -1.), &add(&b11, &b12, 1.), leaf)?;
    let m7 = rec(&add(&a12, &a22, -1.), &add(&b21, &b22, 1.), leaf)?;
    let mut c = Matrix::zeros(n, n);
    for i in 0..h {
        for j in 0..h {
            c[(i, j)] = m1[(i, j)] + m4[(i, j)] - m5[(i, j)] + m7[(i, j)];
            c[(i, j + h)] = m3[(i, j)] + m5[(i, j)];
            c[(i + h, j)] = m2[(i, j)] + m4[(i, j)];
            c[(i + h, j + h)] = m1[(i, j)] - m2[(i, j)] + m3[(i, j)] + m6[(i, j)];
        }
    }
    finite(c.as_slice())?;
    Ok(c)
}
