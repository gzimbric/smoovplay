//! Python's `random.Random(seed)` (MT19937, seeded by `init_by_array`), for the draws Ps5SimSource makes: the same
//! seed gives the same hitches as framegen/io.py.

pub struct PyRandom {
    mt: [u32; 624],
    i: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> PyRandom {
        let mut r = PyRandom { mt: [0; 624], i: 624 };
        r.init_genrand(19650218);
        // Python's seed(int): the absolute value as 32-bit words, least significant first (at least one)
        let mut key = vec![(seed & 0xffff_ffff) as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let n = 624;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..n.max(key.len()) {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                r.mt[0] = r.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..n - 1 {
            let prev = r.mt[i - 1] ^ (r.mt[i - 1] >> 30);
            r.mt[i] = (r.mt[i] ^ prev.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                r.mt[0] = r.mt[n - 1];
                i = 1;
            }
        }
        r.mt[0] = 0x8000_0000;
        r.i = 624;
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.i = 624;
    }

    fn genrand(&mut self) -> u32 {
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
        y ^= y >> 18;
        y
    }

    /// random.random(): [0, 1) with 53 bits.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand() >> 5) as f64;
        let b = (self.genrand() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// random.randrange(n) for 0 < n < 2^32.
    pub fn randrange(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.genrand() >> (32 - k);
            if r < n {
                return r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python() {
        let mut r = PyRandom::new(0);
        assert_eq!([r.random(), r.random(), r.random()], [0.8444218515250481, 0.7579544029403025, 0.420571580830845]);
        let mut r = PyRandom::new(5);
        assert_eq!((0..10).map(|_| r.randrange(6)).collect::<Vec<_>>(), vec![4, 2, 5, 2, 5, 5, 5, 4, 0, 3]);
        assert_eq!(PyRandom::new(123456789).random(), 0.6414006161858726);
    }
}
