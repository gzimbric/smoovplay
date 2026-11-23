//! Opens one fresh tap name from two threads at once, many times, and reports any failure (the create race).
//!     cargo run --release -p sp-tap --example race_stress [-- ROUNDS]
use std::sync::{Arc, Barrier};

fn main() {
    let rounds: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2000);
    let mut failures = std::collections::BTreeMap::<String, usize>::new();
    for k in 0..rounds {
        let n = format!("Local\\sp-rs-race{}", k % 50);
        sp_tap::unlink(&n);
        let b = Arc::new(Barrier::new(2));
        let (b2, n2) = (b.clone(), n.clone());
        let th = std::thread::spawn(move || {
            b2.wait();
            sp_tap::FrameTap::open(&n2).map(|_| ()).map_err(|e| e.to_string())
        });
        b.wait();
        let r1 = sp_tap::FrameTap::open(&n).map(|_| ()).map_err(|e| e.to_string());
        let r2 = th.join().unwrap();
        for r in [r1, r2] {
            if let Err(e) = r {
                *failures.entry(e).or_default() += 1;
            }
        }
        sp_tap::unlink(&n);
    }
    println!("{rounds} rounds: {failures:?}");
}
