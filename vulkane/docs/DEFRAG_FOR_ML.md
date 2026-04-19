# Defragmenting Vulkane Allocations From an ML Library

Target reader: an LLM or engineer integrating Vulkane's sub-allocator into a
training / inference framework where long-lived weight tensors and short-lived
activation tensors share the same VRAM pool. This document describes an
already-shipped API; no feature-gating or version check is required.

## When you need defrag

Fragmentation shows up as a specific failure mode:

- `Allocator::query_budget()` says you have, e.g., 6 GB free on the heap.
- You ask for a 4 GB activation buffer.
- `allocate()` returns `Err(Vk(ERROR_OUT_OF_DEVICE_MEMORY))`.

That's genuine internal fragmentation: total free bytes exceed the request,
but no single contiguous region is large enough. Planned eviction (freeing
specific tensors) and reactive fallback (re-trying with a smaller allocation)
neither fixes the root cause. Compacting live allocations to close the gaps
does.

Not every scenario needs defrag:

- **Linear / bump pools** — reset the whole pool in one call
  ([`Allocator::reset_pool`]). The transient/per-step pool pattern is almost
  always the right shape for activations. Only reach for defrag when you have
  long-lived, irregular allocations in a general-purpose (TLSF) pool.
- **Dedicated allocations** — backing resources that each live in their own
  `VkDeviceMemory` never fragment the allocator. They also can't be defragged
  (they'd need a new Vulkan allocation anyway). Weights that are ≥ half the
  block size fall into this category automatically.

If your ML framework has a clear weights-vs-activations split, consider:
- Weights → a dedicated **FreeList** custom pool (defragmentable).
- Activations → a **Linear** custom pool per training step, reset on step
  boundary (no defrag needed).

## The API

Vulkane exposes a *planned* defragmentation API: the allocator computes what
should move, the app issues GPU work to perform the moves, then the allocator
commits the new layout. This gives you full control over command-stream
scheduling, fences, and resource rebinding — critical in an ML framework
where you already have your own GPU queue / stream abstraction.

Three types, two methods, from [`vulkane::safe`](https://docs.rs/vulkane):

```rust
use vulkane::safe::{
    Allocator, PoolHandle,
    DefragmentationMove,  // one src->dst entry
    DefragmentationPlan,  // the complete plan returned by build_...
};

let plan: DefragmentationPlan = allocator.build_defragmentation_plan(pool_handle);
// ... issue GPU copies and resource rebinds ...
allocator.apply_defragmentation_plan(plan);
```

### `build_defragmentation_plan(pool: PoolHandle) -> DefragmentationPlan`

Snapshots every live allocation in the pool, sorts them by
`(block_index, offset)`, and computes a compacted target layout: everything
packed to the start of block 0 with 256-byte alignment between allocations.
The returned plan contains:

- `plan.moves: Vec<DefragmentationMove>` — only the allocations whose
  `(memory, offset)` will change. Each entry has:
  - `allocation_id: u64` — stable id; matches `Allocation::id()`.
  - `user_data: u64` — whatever you passed to
    `AllocationCreateInfo::user_data` at allocation time. Use this to map
    back to your framework's tensor handle / buffer wrapper.
  - `size: u64`
  - `src_memory, src_offset` — where the allocation currently lives.
  - `dst_memory, dst_offset` — where it needs to end up.
- `plan.bytes_freed: u64` — estimate of contiguous bytes you'll reclaim.
- `plan.total_layout()` — read-only view of the complete post-defrag layout
  (includes unchanged allocations). Rarely needed; diagnostic only.

The plan is pure metadata. Nothing on the GPU has moved yet.

**Scope**: only `FreeList` (TLSF) custom pools. Linear pools return an empty
plan — use `reset_pool` instead. Dedicated allocations never participate.
Default per-memory-type pools are not currently targetable; if you want defrag,
`create_pool(...)` with `strategy: AllocationStrategy::FreeList` and route
defragmentable allocations through it.

### `apply_defragmentation_plan(plan: DefragmentationPlan)`

Walks the plan, re-allocates each entry at its target offset inside the
allocator's internal TLSF state, and rewrites every live `Allocation`'s
`(memory, offset)` in place. After this call:

- Every clone of every affected `Allocation` — including any copies your
  framework is holding — returns the new `memory()` / `offset()` on its next
  accessor call. This is atomic per allocation (internal `Mutex` on the
  location).
- The TLSF side-table is rebuilt from scratch based on the plan's layout.
- `Allocation::id()` is stable across the move. Use it as a primary key if
  you're maintaining your own tensor-to-allocation map.

**Preconditions** (the allocator trusts you on these — violating them is a
use-after-free):
1. The GPU copy commands for every `plan.moves` entry have been recorded and
   *completed*. A `Fence::wait` or `vkQueueWaitIdle` is the simplest way to
   guarantee this.
2. The old `Buffer`/`Image` objects backed by the moved allocations have been
   destroyed or rebound. The driver treats `vkBindBufferMemory` as a
   one-shot: once a buffer is bound, you cannot rebind it — you destroy and
   re-create it at the new offset.
3. No other thread is racing on the moved allocations.

Unmoved allocations are untouched (both `src` and `dst` are equal, so no
GPU work is needed for them, but their Rust-side state is rewritten to
point to the rebuilt TLSF bookkeeping anyway).

## Worked example: compacting a TLSF pool of long-lived tensors

Here's the minimal skeleton a framework would build around the two calls.
Error handling elided for brevity; real code should propagate allocator errors.

```rust
use vulkane::safe::{
    AccessFlags, AllocationCreateInfo, AllocationStrategy, AllocationUsage,
    Allocator, Buffer, BufferCopy, BufferCreateInfo, BufferUsage,
    CommandPool, DefragmentationPlan, Device, Fence, PipelineStage,
    PoolCreateInfo, PoolHandle, Queue,
};

/// Tensor handle the ML framework holds onto. Keeps the Allocation
/// (so the slot is refcounted) and the current Buffer (which is
/// invalidated across a defrag cycle and must be recreated).
struct Tensor {
    id: u64,                    // framework-side key
    allocation: vulkane::safe::Allocation,
    buffer: Buffer,
}

fn defrag_tensor_pool(
    device: &Device,
    queue: &Queue,
    queue_family: u32,
    allocator: &Allocator,
    pool: PoolHandle,
    tensors: &mut [Tensor],
) -> Result<(), vulkane::safe::Error> {
    // 1. Build the plan. No GPU work yet; pure bookkeeping.
    let plan = allocator.build_defragmentation_plan(pool);
    if plan.moves.is_empty() {
        return Ok(()); // already compact
    }

    // 2. Record a command buffer that copies each moved allocation from
    //    its old (src) location to its new (dst) location. We use the
    //    pre-existing Buffer for the src and a fresh Buffer for the dst.
    let cmd_pool = CommandPool::new(device, queue_family)?;
    let mut cmd = cmd_pool.allocate_primary()?;

    // Collect new buffers so they outlive the submission.
    let mut new_buffers: Vec<(u64 /* alloc_id */, Buffer)> = Vec::new();

    {
        let mut rec = cmd.begin()?;
        for mv in &plan.moves {
            // Find the owning tensor via user_data (or allocation_id).
            let tensor_idx = tensors
                .iter()
                .position(|t| t.allocation.id() == mv.allocation_id)
                .expect("plan references a live allocation we don't know about");

            // Create a new Buffer bound to (dst_memory, dst_offset).
            // We can't just rebind the old buffer — Vulkan binds are
            // one-shot. Bind a fresh buffer at the destination.
            let new_buffer = Buffer::new(
                device,
                BufferCreateInfo {
                    size: mv.size,
                    usage: BufferUsage::STORAGE_BUFFER
                        | BufferUsage::TRANSFER_SRC
                        | BufferUsage::TRANSFER_DST,
                },
            )?;
            // Use the dispatch-table binding directly since the
            // allocator owns the VkDeviceMemory lifetime.
            let bind = device
                .dispatch()
                .vkBindBufferMemory
                .ok_or(vulkane::safe::Error::MissingFunction("vkBindBufferMemory"))?;
            // Safety: dst_memory is owned by the allocator, valid until
            // we call free or drop the allocator; new_buffer is a fresh
            // handle we control.
            let r = unsafe {
                bind(device.raw(), new_buffer.raw(), mv.dst_memory, mv.dst_offset)
            };
            vulkane::safe::check(r)?;

            rec.copy_buffer(
                &tensors[tensor_idx].buffer,
                &new_buffer,
                &[BufferCopy {
                    src_offset: mv.src_offset,
                    dst_offset: 0,
                    size: mv.size,
                }],
            );
            new_buffers.push((mv.allocation_id, new_buffer));
        }
        // Make every copy visible before any subsequent shader read.
        rec.memory_barrier(
            PipelineStage::TRANSFER,
            PipelineStage::COMPUTE_SHADER | PipelineStage::VERTEX_SHADER,
            AccessFlags::TRANSFER_WRITE,
            AccessFlags::SHADER_READ,
        );
        rec.end()?;
    }

    // 3. Submit + wait. No work can see the old layout after this point.
    let fence = Fence::new(device)?;
    queue.submit(&[&cmd], Some(&fence))?;
    fence.wait(u64::MAX)?;

    // 4. Commit the plan. Every Allocation clone now returns the new
    //    (memory, offset) via its accessors.
    allocator.apply_defragmentation_plan(plan);

    // 5. Swap each tensor's Buffer to the new one we created and bound.
    //    The old Buffer's Drop destroys it (its VkDeviceMemory is still
    //    owned by the allocator — we never owned that side).
    for (alloc_id, new_buffer) in new_buffers {
        if let Some(t) = tensors.iter_mut().find(|t| t.allocation.id() == alloc_id) {
            t.buffer = new_buffer;
        }
    }

    Ok(())
}
```

The key observations:
- `Allocation::id()` is the stable primary key — survives the defrag and is
  unique within an allocator. `user_data` on the move is also surfaced if
  your framework prefers a domain-specific handle (cast of an `Arc<Tensor>`
  raw pointer, a slot index, etc. — set it at allocation time via
  `AllocationCreateInfo::user_data`).
- You create new `Buffer` handles bound to the new `(dst_memory, dst_offset)`.
  The old ones must be destroyed before the allocator's internal state is
  updated, or immediately after — but their memory lifetime is the
  allocator's, not the buffer's.
- The fence wait between step 3 and step 4 is non-negotiable. Committing the
  plan while copies are still in flight means subsequent work sees the new
  layout with undefined contents.

## Integrating with planned eviction and reactive OOM

A robust ML scheduler will layer defrag under its existing eviction paths:

1. **Budget-based planning** (cheap, first-line):
   `allocator.vram_budget()` and `allocator.vram_used()` (see
   `VK_EXT_memory_budget` — auto-enabled by Vulkane when the driver
   supports it). If projected usage would push past, say, 85 % of budget,
   start evicting LRU tensors preemptively.

2. **Proactive pressure callbacks**:
   `allocator.register_pressure_callback(threshold, hysteresis, cb)` and
   `allocator.would_fit(size, memory_type_index)` let the scheduler register
   a threshold once and be notified (including *before* an attempted
   allocation crosses it) instead of polling.

3. **Defrag** (mid-cost, fires on fragmentation-specific failures): when
   `would_fit` says the projection fits but `allocate` actually fails, or
   when free-region count grows pathologically without usage growing,
   schedule a defrag pass. Targets: the dedicated weights pool, not the
   per-step activation pools (those get reset).

4. **Reactive eviction** (last resort): if defrag + eviction both fail,
   your framework's backstop "just free something" path runs.

Defrag is O(live_allocations · log) in bookkeeping plus the actual GPU copy
cost — proportional to moved bytes. It's not cheap, but it's bounded and
predictable, and unlike allocator-cycle eviction it never discards user
state.

## Edge cases worth knowing

- **Mapped pointers stay valid across the move**. The pool's block-level
  persistent mapping does not change; the allocation's new offset is
  applied inside the same mapped range. Code that captured the raw
  `*mut c_void` from `Allocation::mapped_ptr()` *before* defrag must
  re-read it after — the pointer's base may be the same but the effective
  address is `base + new_offset`.
- **Linear pools return an empty plan** and are a no-op. Call `reset_pool`.
- **The plan is pool-scoped**. Defrag runs one pool at a time. If you want
  to compact multiple TLSF pools, issue the GPU work independently and
  apply the plans one at a time.
- **Thread safety**: the plan-build and plan-apply methods take `&self`
  and lock internally. You must still prevent other threads from freeing
  or allocating from the pool between `build_...` and `apply_...` if you
  care about the `bytes_freed` estimate staying accurate. A second
  allocation between build and apply doesn't corrupt state (apply operates
  on the live set at apply time), but the plan's moves may no longer be
  the optimal layout.

## Where to look in the crate

- [`vulkane::safe::Allocator::build_defragmentation_plan`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.build_defragmentation_plan)
- [`vulkane::safe::Allocator::apply_defragmentation_plan`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.apply_defragmentation_plan)
- [`vulkane::safe::DefragmentationPlan`](https://docs.rs/vulkane/latest/vulkane/safe/struct.DefragmentationPlan.html)
- [`vulkane::safe::DefragmentationMove`](https://docs.rs/vulkane/latest/vulkane/safe/struct.DefragmentationMove.html)
- Companion budget / pressure API:
  [`Allocator::vram_budget`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.vram_budget),
  [`vram_used`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.vram_used),
  [`register_pressure_callback`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.register_pressure_callback),
  [`would_fit`](https://docs.rs/vulkane/latest/vulkane/safe/struct.Allocator.html#method.would_fit).
