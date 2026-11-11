use numerical_algorithms::*;

fn mat(r: usize, c: usize, x: &[f64]) -> Matrix {
    Matrix::new(r, c, x.to_vec()).unwrap()
}
fn close(x: &[f64], y: &[f64], tol: f64) {
    assert_eq!(x.len(), y.len());
    for (a, b) in x.iter().zip(y) {
        assert!(
            (a - b).abs() <= tol * (1. + b.abs()),
            "{a} != {b}, tolerance {tol}"
        );
    }
}
fn same(a: &Matrix, b: &Matrix, t: f64) {
    assert_eq!((a.rows(), a.cols()), (b.rows(), b.cols()));
    close(a.as_slice(), b.as_slice(), t);
}
fn sample() -> Matrix {
    mat(3, 3, &[0., 2., 1., 3., -1., 2., 1., 1., 4.])
}
fn generated(n: usize) -> Matrix {
    let mut a = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 17 + j * 11 + 3) as f64).sin();
        }
        a[(i, i)] += n as f64;
    }
    a
}
fn truth(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i + 2) as f64).cos()).collect()
}

#[test]
fn strassen_odd_sizes_and_leaf_sizes() {
    for n in 1..10 {
        let a = generated(n);
        let b = a.transpose();
        for leaf in [1, 2, 8] {
            same(
                &fast::strassen(&a, &b, leaf).unwrap(),
                &a.mul(&b).unwrap(),
                1e-11,
            );
        }
    }
}

