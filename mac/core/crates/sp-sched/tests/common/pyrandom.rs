//! Python's random.Random (MT19937, seeded from an int as CPython does), so the ported tests play exactly the games
//! the Python tests do.

pub struct PyRandom {
    mt: [u32; 624],
    i: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> PyRandom {
        let mut key: Vec<u32> = Vec::new();
        let mut s = seed;
        loop {
            key.push(s as u32);
            s >>= 32;
            if s == 0 {
                break;
            }
        }
        let mut r = PyRandom { mt: [0; 624], i: 624 };
        r.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let prev = r.mt[i - 1];
            r.mt[i] = (r.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
        }
        r.mt[0] = 0x8000_0000;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            let p = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(p ^ (p >> 30)).wrapping_add(i as u32);
        }
        self.i = 624;
    }

    fn next_u32(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.mt[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// random.random().
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// random.uniform(a, b).
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// A standard normal (Box-Muller, not numpy's), for the tests that only need some jitter.
    pub fn normal(&mut self) -> f64 {
        let u = self.random().max(1e-300);
        let v = self.random();
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    #[test]
    fn matches_cpython() {
        let want: [(u64, [f64; 3], f64); 5] = [
            (0, [0.8444218515250481, 0.7579544029403025, 0.420571580830845], 0.8794583751464817),
            (1, [0.13436424411240122, 0.8474337369372327, 0.763774618976614], 0.8775345128697108),
            (2, [0.9560342718892494, 0.9478274870593494, 0.05655136772680869], 0.7924359975794608),
            (7, [0.32383276483316237, 0.15084917392450192, 0.6509344730398537], 0.7862181433337714),
            ((1 << 40) + 5, [0.5043802970418443, 0.2686044399723282, 0.9257865475671585], 1.015479619646226),
        ];
        for (seed, xs, u) in want {
            let mut r = PyRandom::new(seed);
            assert_eq!([r.random(), r.random(), r.random()], xs, "seed {seed}");
            assert_eq!(r.uniform(0.75, 1.25), u);
        }
    }
}
