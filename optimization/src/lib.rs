//! Smooth unconstrained minimization with an analytic gradient.

mod direction;
mod line_search;

use direction::Direction;
use line_search::{strong_wolfe, Point};
use std::fmt;

/// How the search direction is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Bfgs,
    Lbfgs {
        memory: usize,
    },
    /// Nonlinear conjugate gradient with the Polak–Ribière+ coefficient.
    ConjugateGradient,
}

/// Limits for the outer iterations and the line search.
#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// Stop when the largest absolute gradient component is at most this value.
    pub gradient_tolerance: f64,
    /// Maximum number of accepted steps.
    pub max_iterations: usize,
    /// Maximum objective evaluations per line-search attempt.
    pub max_line_search_evaluations: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-6,
            max_iterations: 1_000,
            max_line_search_evaluations: 50,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Converged,
    IterationLimit,
    LineSearchFailed,
}

/// The last accepted iterate, even if the method stopped before converging.
#[derive(Debug, Clone)]
pub struct Solution {
    pub x: Vec<f64>,
    pub value: f64,
    pub gradient: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub status: Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputError {
    InvalidStart,
    InvalidOptions,
    ZeroMemory,
    InvalidInitialEvaluation,
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidStart => "the initial vector must be nonempty and finite",
            Self::InvalidOptions => "tolerance must be finite and positive and the line search needs an evaluation budget",
            Self::ZeroMemory => "L-BFGS memory must be positive",
            Self::InvalidInitialEvaluation => "the initial objective and all gradient components must be finite",
        };
        f.write_str(message)
    }
}

impl std::error::Error for InputError {}

/// Minimize an objective from `x0`. The callback returns the objective value
/// and writes every component of the gradient into its second argument.
/// At trial points outside the objective's domain, return a nonfinite value.
pub fn minimize(
    method: Method,
    x0: &[f64],
    options: Options,
    mut objective: impl FnMut(&[f64], &mut [f64]) -> f64,
) -> Result<Solution, InputError> {
    validate(method, x0, options)?;
    let mut evaluations = 0;
    let mut evaluate = |x: &[f64], gradient: &mut [f64]| {
        evaluations += 1;
        objective(x, gradient)
    };
    let mut point =
        Point::evaluate(x0.to_vec(), &mut evaluate).ok_or(InputError::InvalidInitialEvaluation)?;

    let c2 = if method == Method::ConjugateGradient {
        0.1
    } else {
        0.9
    };
    let mut direction = Direction::new(method);
    let mut iterations = 0;
    let status = loop {
        if point
            .gradient
            .iter()
            .all(|g| g.abs() <= options.gradient_tolerance)
        {
            break Status::Converged;
        }
        if iterations == options.max_iterations {
            break Status::IterationLimit;
        }
        let Some(next) = take_step(&point, &mut direction, c2, options, &mut evaluate) else {
            break Status::LineSearchFailed;
        };
        point = next;
        iterations += 1;
    };

    Ok(Solution {
        x: point.x,
        value: point.value,
        gradient: point.gradient,
        iterations,
        evaluations,
        status,
    })
}

fn validate(method: Method, x0: &[f64], options: Options) -> Result<(), InputError> {
    if x0.is_empty() || x0.iter().any(|x| !x.is_finite()) {
        return Err(InputError::InvalidStart);
    }
    if !options.gradient_tolerance.is_finite()
        || options.gradient_tolerance <= 0.0
        || options.max_line_search_evaluations == 0
    {
        return Err(InputError::InvalidOptions);
    }
    if method == (Method::Lbfgs { memory: 0 }) {
        return Err(InputError::ZeroMemory);
    }
    Ok(())
}

/// Search along the proposed direction. Restart with steepest descent if needed.
fn take_step(
    point: &Point,
    direction: &mut Direction,
    c2: f64,
    options: Options,
    evaluate: &mut impl FnMut(&[f64], &mut [f64]) -> f64,
) -> Option<Point> {
    let steepest = negate(&point.gradient);
    let mut p = direction.propose(&point.gradient);
    let slope = dot(&p, &point.gradient);
    if !slope.is_finite() || slope >= 0.0 {
        direction.reset();
        p.clone_from(&steepest);
    }

    let search = |p: &[f64], evaluate: &mut _| {
        strong_wolfe(point, p, c2, options.max_line_search_evaluations, evaluate)
    };
    let next = match search(&p, evaluate) {
        Some(next) => next,
        None if p != steepest => {
            direction.reset();
            p = steepest;
            search(&p, evaluate)?
        }
        None => return None,
    };
    direction.update(point, &next, p);
    Some(next)
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(x, y)| x * y).sum()
}

fn negate(x: &[f64]) -> Vec<f64> {
    x.iter().map(|x| -x).collect()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.iter().zip(y).map(|(x, y)| x - y).collect()
}
