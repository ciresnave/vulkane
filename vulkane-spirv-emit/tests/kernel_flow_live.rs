// SPDX-License-Identifier: MIT OR Apache-2.0
//! ROADMAP row 8g on a live device: nested loops with `Break`, `Continue` and an early `Return`
//! (`tests/support/flow_kernel.rs`), lowered by `SpirV::lower_kernel` and compared per thread with
//! a host reference. Thread 7 returns before its store, so its cell must still hold the sentinel
//! (a never-wrote check that a wrongly lowered `Return` would fail).
//!
//! Without a device the test follows `VULKANE_REQUIRE_DEVICE` like every other live test.

#[path = "../../vulkane/tests/common/mod.rs"]
#[allow(dead_code, unreachable_pub, missing_docs)]
mod common;
#[path = "support/flow_kernel.rs"]
mod flow_kernel;
#[path = "support/gpu.rs"]
mod gpu;

use unpopped::backend::Backend;
use vulkane_spirv_emit::SpirV;

#[test]
fn nested_loops_break_continue_and_return_match_the_host_on_a_device() {
    common::require_serialization_lock();
    let Some(gpu) = gpu::Gpu::open("kernel_flow") else {
        return; // skip already declared
    };
    let g = SpirV
        .lower_kernel(&flow_kernel::flow_kernel())
        .expect("the SPIR-V backend lowers the flow kernel");
    let words = g.words().expect("words").to_vec();
    let got = gpu.run(&g.name, &words, &[&[0u32; 256]]);
    let mut returned = 0;
    for (t, &g) in got.iter().enumerate() {
        match flow_kernel::expected(t as u32) {
            Some(w) => assert_eq!(g, w, "thread {t}"),
            None => {
                assert_eq!(g, gpu::SENTINEL, "thread {t} returned before its store");
                returned += 1;
            }
        }
    }
    assert_eq!(returned, 1, "exactly one thread takes the early return");
}
