//! The few numpy and Python behaviours the reports depend on, reproduced exactly so the Rust text matches the
//! Python reference character for character: numpy's pairwise summation (np.sum, np.mean), np.median, np.percentile
//! (method "linear"), Python's round() (half to even) and its float formatting ("nan", not "NaN").

/// numpy's pairwise sum of a float64 array (numpy/_core/src/umath/loops_utils.h.src, PW_BLOCKSIZE 128).
pub fn sum(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 8 {
        let mut r = 0.0;
        for &x in xs {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&xs[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += xs[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += xs[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        sum(&xs[..n2]) + sum(&xs[n2..])
    }
}

/// np.mean, which is NaN for an empty list (numpy warns and returns nan).
pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    sum(xs) / xs.len() as f64
}

fn sorted(xs: &[f64]) -> Vec<f64> {
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s
}

/// np.median (NaN if empty or if any value is NaN).
pub fn median(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 || xs.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let s = sorted(xs);
    if n % 2 == 1 {
        mean(&s[n / 2..n / 2 + 1])
    } else {
        mean(&s[n / 2 - 1..n / 2 + 1])
    }
}

/// np.percentile(xs, p), method "linear", as numpy 2 computes it (its _lerp included).
pub fn percentile(xs: &[f64], p: f64) -> f64 {
    let n = xs.len();
    if n == 0 || xs.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let s = sorted(xs);
    let q = p / 100.0;
    let vi = (n - 1) as f64 * q;
    let (pi, ni, prev) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1, -1.0)
    } else {
        let f = vi.floor();
        (f as usize, f as usize + 1, f)
    };
    let gamma = vi - prev;
    lerp(s[pi], s[ni], gamma)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let d = b - a;
    if t >= 0.5 { b - d * (1.0 - t) } else { a + d * t }
}

/// np.percentile on a float32 array, in float32 arithmetic because numpy keeps the array's type.
pub fn percentile_f32(xs: &[f32], p: f64) -> f32 {
    let n = xs.len();
    if n == 0 || xs.iter().any(|x| x.is_nan()) {
        return f32::NAN;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = p / 100.0;
    let vi = (n - 1) as f64 * q;
    let (pi, ni, prev) = if vi >= (n - 1) as f64 {
        (n - 1, n - 1, -1.0)
    } else {
        let f = vi.floor();
        (f as usize, f as usize + 1, f)
    };
    let t = (vi - prev) as f32;
    let (a, b) = (s[pi], s[ni]);
    let d = b - a;
    if t >= 0.5 { b - d * (1.0 - t) } else { a + d * t }
}

/// Python's round(x) for a float, to the nearest integer with halves to even.
pub fn round(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// Python's f"{x:.{prec}f}".
pub fn f(x: f64, prec: usize) -> String {
    if x.is_nan() {
        "nan".into()
    } else if x.is_infinite() {
        if x > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        format!("{x:.prec$}")
    }
}

/// Python's f"{x:{width}.{prec}f}" (right-aligned).
pub fn fw(x: f64, width: usize, prec: usize) -> String {
    format!("{:>width$}", f(x, prec))
}

/// The last `n` items (Python's xs[-n:]).
pub fn tail<T>(xs: &[T], n: usize) -> &[T] {
    &xs[xs.len().saturating_sub(n)..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy_on_known_values() {
        let xs = [1.0, 2.0, 3.0, 4.0, 10.0];
        assert_eq!(percentile(&xs, 50.0), 3.0);
        assert_eq!(percentile(&xs, 95.0), 8.799999999999999); // numpy: 8.799999999999999
        assert_eq!(percentile(&xs, 100.0), 10.0);
        assert_eq!(median(&[1.0, 2.0, 4.0, 7.0]), 3.0);
    }

    #[test]
    fn python_formatting() {
        assert_eq!(f(f64::NAN, 2), "nan");
        assert_eq!(f(0.125, 2), "0.12");
        assert_eq!(f(2.5, 0), "2");
        assert_eq!(fw(1.23456, 6, 2), "  1.23");
        assert_eq!(round(2.5), 2);
        assert_eq!(round(3.5), 4);
    }
}
