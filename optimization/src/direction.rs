use crate::{axpy, difference, dot, negate, Method, Point};
use std::collections::VecDeque;

pub(crate) enum Direction {
    Bfgs {
        inverse: Vec<Vec<f64>>,
    },
    Lbfgs {
        memory: usize,
        pairs: VecDeque<Pair>,
    },
    ConjugateGradient {
        next: Vec<f64>,
    },
}

impl Direction {
    pub(crate) fn new(method: Method) -> Self {
        match method {
            Method::Bfgs => Self::Bfgs {
                inverse: Vec::new(),
            },
            Method::Lbfgs { memory } => Self::Lbfgs {
                memory,
                pairs: VecDeque::new(),
            },
            Method::ConjugateGradient => Self::ConjugateGradient { next: Vec::new() },
        }
    }

    pub(crate) fn reset(&mut self) {
        match self {
            Self::Bfgs { inverse } => inverse.clear(),
            Self::Lbfgs { pairs, .. } => pairs.clear(),
            Self::ConjugateGradient { next } => next.clear(),
        }
    }

    pub(crate) fn propose(&self, gradient: &[f64]) -> Vec<f64> {
        match self {
            Self::Bfgs { inverse } if !inverse.is_empty() => {
                inverse.iter().map(|row| -dot(row, gradient)).collect()
            }
            Self::Lbfgs { pairs, .. } if !pairs.is_empty() => two_loop(pairs, gradient),
            Self::ConjugateGradient { next } if !next.is_empty() => next.clone(),
            _ => negate(gradient),
        }
    }

    pub(crate) fn update(&mut self, current: &Point, trial: &Point, p: Vec<f64>) {
        if let Self::ConjugateGradient { next } = self {
            *next = polak_ribiere(&current.gradient, &trial.gradient, &p);
            return;
        }

        let Some(pair) = Pair::new(current, trial) else {
            return;
        };
        match self {
            Self::Bfgs { inverse } => update_inverse(inverse, &pair),
            Self::Lbfgs { memory, pairs } => {
                if pairs.len() == *memory {
                    pairs.pop_front();
                }
                pairs.push_back(pair);
            }
            Self::ConjugateGradient { .. } => unreachable!(),
        }
    }
}

/// Displacement, gradient difference, and their curvature information.
pub(crate) struct Pair {
    s: Vec<f64>,
    y: Vec<f64>,
    rho: f64,
    scale: f64,
}

impl Pair {
    fn new(current: &Point, trial: &Point) -> Option<Self> {
        let s = difference(&trial.x, &current.x);
        let y = difference(&trial.gradient, &current.gradient);
        let sy = dot(&s, &y);
        let yy = dot(&y, &y);
        let threshold = 1e-12 * dot(&s, &s).sqrt() * yy.sqrt();
        let rho = sy.recip();
        let scale = sy / yy;
        if sy > threshold && rho.is_finite() && rho > 0.0 && scale.is_finite() && scale > 0.0 {
            Some(Self { s, y, rho, scale })
        } else {
            None
        }
    }
}

/// Apply the expanded inverse BFGS formula in quadratic time.
fn update_inverse(inverse: &mut Vec<Vec<f64>>, pair: &Pair) {
    let Pair { s, y, rho, .. } = pair;
    if inverse.is_empty() {
        let n = s.len();
        *inverse = vec![vec![0.0; n]; n];
        for (i, row) in inverse.iter_mut().enumerate() {
            row[i] = 1.0;
        }
    }
    let hy: Vec<_> = inverse.iter().map(|row| dot(row, y)).collect();
    let factor = 1.0 + rho * dot(y, &hy);
    for (i, row) in inverse.iter_mut().enumerate() {
        for (j, hij) in row.iter_mut().enumerate() {
            *hij += factor * (rho * s[i]) * s[j] - (rho * s[i]) * hy[j] - (rho * s[j]) * hy[i];
        }
    }
}

/// Apply the implicit inverse Hessian with the L-BFGS two-loop recursion.
fn two_loop(pairs: &VecDeque<Pair>, gradient: &[f64]) -> Vec<f64> {
    let mut q = gradient.to_vec();
    let mut alphas = Vec::with_capacity(pairs.len());
    for pair in pairs.iter().rev() {
        let alpha = pair.rho * dot(&pair.s, &q);
        axpy(&mut q, -alpha, &pair.y);
        alphas.push(alpha);
    }
    let scale = pairs.back().expect("history is nonempty").scale;
    let mut r: Vec<_> = q.iter().map(|qi| scale * qi).collect();
    for (pair, alpha) in pairs.iter().zip(alphas.into_iter().rev()) {
        let beta = pair.rho * dot(&pair.y, &r);
        axpy(&mut r, alpha - beta, &pair.s);
    }
    negate(&r)
}

/// Compute PR+ and restart when the direction has too little descent.
fn polak_ribiere(old: &[f64], new: &[f64], p: &[f64]) -> Vec<f64> {
    let mut next = negate(new);
    let beta = dot(new, &difference(new, old)) / dot(old, old);
    if beta.is_finite() && beta > 0.0 {
        axpy(&mut next, beta, p);
        let slope = dot(new, &next);
        if !slope.is_finite() || slope >= -1e-3 * dot(new, new) {
            return negate(new);
        }
    }
    next
}
