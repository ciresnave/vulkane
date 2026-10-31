//! Throwaway repro for vulkane#130 (not for merge): N threads, each creating and
//! dropping its own Instance + Device through the same helper the live tests use.
//! RACE_MODE selects what is serialized:
//!   plain      nothing
//!   lock_new   a global mutex around creation only
//!   lock_all   a global mutex around creation and drop
//!   keep       no lock; every thread holds its device until all have created one
mod common;

use std::sync::{Arc, Barrier, Mutex};

static BIG: Mutex<()> = Mutex::new(());

#[test]
fn race() {
    let mode = std::env::var("RACE_MODE").unwrap_or_else(|_| "plain".into());
    let threads: usize = std::env::var("RACE_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let iters: usize = std::env::var("RACE_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
    let barrier = Arc::new(Barrier::new(threads));
    let hs: Vec<_> = (0..threads)
        .map(|_| {
            let mode = mode.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..iters {
                    match mode.as_str() {
                        "lock_new" => {
                            let r = {
                                let _g = BIG.lock().unwrap();
                                common::compute_device("race")
                            };
                            drop(r);
                        }
                        "lock_all" => {
                            let _g = BIG.lock().unwrap();
                            drop(common::compute_device("race"));
                        }
                        _ => drop(common::compute_device("race")),
                    }
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
}
