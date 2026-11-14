use crate::dot;

const C1: f64 = 1e-4;

pub(crate) struct Point {
    pub(crate) x: Vec<f64>,
    pub(crate) value: f64,
    pub(crate) gradient: Vec<f64>,
}

impl Point {
    pub(crate) fn evaluate(
        x: Vec<f64>,
        evaluate: &mut impl FnMut(&[f64], &mut [f64]) -> f64,
    ) -> Option<Self> {
        if x.iter().any(|xi| !xi.is_finite()) {
            return None;
        }
        let mut gradient = vec![f64::NAN; x.len()];
        let value = evaluate(&x, &mut gradient);
        if !value.is_finite() || gradient.iter().any(|gi| !gi.is_finite()) {
            return None;
        }
        Some(Self { x, value, gradient })
    }
}

struct Search<'a> {
    start: &'a Point,
    direction: &'a [f64],
    slope: f64,
    max_slope: f64,
}

impl Search<'_> {
    fn trial(
        &self,
        alpha: f64,
        evaluate: &mut impl FnMut(&[f64], &mut [f64]) -> f64,
    ) -> Option<(Point, f64)> {
        let x: Vec<_> = self
            .start
            .x
            .iter()
            .zip(self.direction)
            .map(|(xi, pi)| xi + alpha * pi)
            .collect();
        let point = Point::evaluate(x, evaluate)?;
        let slope = dot(&point.gradient, self.direction);
        let armijo = self.start.value + C1 * alpha * self.slope;
        if point.value <= armijo && slope.is_finite() {
            Some((point, slope))
        } else {
            None
        }
    }

    /// Bisect a bracket until both Wolfe conditions hold or the budget runs out.
    fn zoom(
        &self,
        mut low: f64,
        mut high: f64,
        mut low_value: f64,
        budget: usize,
        evaluate: &mut impl FnMut(&[f64], &mut [f64]) -> f64,
    ) -> Option<Point> {
        for _ in 0..budget {
            let alpha = low + 0.5 * (high - low);
            if alpha == low || alpha == high {
                break;
            }
            match self.trial(alpha, evaluate) {
                Some((point, slope)) if point.value <= low_value => {
                    if slope.abs() <= self.max_slope {
                        return Some(point);
                    }
                    if slope.is_sign_positive() == (high > low) {
                        high = low;
                    }
                    low = alpha;
                    low_value = point.value;
                }
                _ => high = alpha,
            }
        }
        None
    }
}

/// Expand the step from one until it succeeds or brackets a candidate.
pub(crate) fn strong_wolfe(
    start: &Point,
    direction: &[f64],
    c2: f64,
    budget: usize,
    evaluate: &mut impl FnMut(&[f64], &mut [f64]) -> f64,
) -> Option<Point> {
    let slope = dot(&start.gradient, direction);
    if !slope.is_finite() || slope >= 0.0 {
        return None;
    }
    let search = Search {
        start,
        direction,
        slope,
        max_slope: -c2 * slope,
    };
    let mut alpha = 1.0;
    let mut previous_alpha = 0.0;
    let mut previous_value = start.value;
    for remaining in (0..budget).rev() {
        let (point, slope) = match search.trial(alpha, evaluate) {
            Some((point, slope)) if point.value <= previous_value => (point, slope),
            _ => return search.zoom(previous_alpha, alpha, previous_value, remaining, evaluate),
        };
        if slope.abs() <= search.max_slope {
            return Some(point);
        }
        if slope >= 0.0 {
            return search.zoom(alpha, previous_alpha, point.value, remaining, evaluate);
        }
        previous_alpha = alpha;
        previous_value = point.value;
        alpha *= 2.0;
        if !alpha.is_finite() {
            break;
        }
    }
    None
}
