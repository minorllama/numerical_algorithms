use optimizers::{minimize, InputError, Method, Options, Status};

fn all_methods_minimize_a_coupled_quadratic() {
    for method in [
        Method::Bfgs,
        Method::Lbfgs { memory: 1 },
        Method::ConjugateGradient,
    ] {
        let result = minimize(method, &[7.0, 5.0], Options::default(), |x, g| {
            let a = x[0] - 1.0;
            let b = x[1] + 2.0;
            g[0] = 3.0 * a + b;
            g[1] = a + 9.0 * b;
            0.5 * (a * g[0] + b * g[1])
        })
        .unwrap();
        println!("{:?}", result);
        println!("{:?} {:?} {:?}\n\n", result.status, Status::Converged, "{method:?}");
        assert_eq!(result.status, Status::Converged, "{method:?}");
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] + 2.0).abs() < 1e-6);
    }
}

fn main(){
    all_methods_minimize_a_coupled_quadratic();
}
