//! Two-Level Segregated Fit (TLSF) free-list allocator for one
//! `VkDeviceMemory` block.
//!
//! Based on the canonical TLSF algorithm by M. Masmano, I. Ripoll, A. Crespo,
//! and J. Real (2004). TLSF is the proven choice for general-purpose GPU
//! sub-allocation: O(1) worst-case allocation and free, low fragmentation
//! (5-15% in steady state on real workloads), and good behaviour for the
//! mixed-size, long-lived allocations that GPU apps produce.
//!
//! ## Block layout
//!
//! We do *not* place the metadata inline in the GPU buffer (the way the C
//! reference implementation does for CPU memory). Instead, we keep a CPU-side
//! `Vec<BlockNode>` of block descriptors, indexed by `BlockId`. Each block
//! descriptor stores its offset within the parent `VkDeviceMemory`, its size,
//! whether it is free or used, the `BlockId` of its physical neighbours
//! (for coalescing), and — when free — the `BlockId` of its next/previous
//! sibling in the matching `(fl, sl)` free bin.
//!
//! ## Bin shape
//!
//! - **First-level (FL):** the highest set bit of the requested size, in the
//!   range `[FL_INDEX_SHIFT .. FL_INDEX_MAX]`. Sizes below `1 << FL_INDEX_SHIFT`
//!   all collapse into the smallest class.
//! - **Second-level (SL):** linear subdivision of the first-level range into
//!   `1 << SL_INDEX_BITS` sub-bins. With `SL_INDEX_BITS = 5` we get 32 sub-bins
//!   per FL class — the same fragmentation/overhead trade-off VMA uses.
//!
//! ## Capacity bounds
//!
//! With `FL_INDEX_SHIFT = 8` and `FL_INDEX_MAX = 32`, a single TLSF block
//! covers requested sizes from 256 bytes up to 4 GiB, which is more than
//! enough for any single Vulkan memory allocation that respects
//! `maxMemoryAllocationSize` on real hardware.

use std::cmp::max;

/// Lower bound on classes — sizes below `1 << FL_INDEX_SHIFT` round up to
/// this value for the purpose of bin selection.
const FL_INDEX_SHIFT: u32 = 8; // 256 bytes
/// Upper bound on classes — anything beyond this is rejected at insert time.
const FL_INDEX_MAX: u32 = 32; // 4 GiB
const FL_INDEX_COUNT: u32 = FL_INDEX_MAX - FL_INDEX_SHIFT + 1;

/// log2 of the number of sub-bins per first-level class.
const SL_INDEX_BITS: u32 = 5;
const SL_INDEX_COUNT: u32 = 1 << SL_INDEX_BITS;

/// Smallest allocation we'll round up to.
const SMALL_BLOCK_SIZE: u64 = 1 << FL_INDEX_SHIFT;

/// `Option<u32>`-style sentinel for "no block".
const NIL: u32 = u32::MAX;

/// One block node in the side-table.
#[derive(Clone, Copy)]
struct BlockNode {
    offset: u64,
    size: u64,
    /// Whether this block is on the free list.
    free: bool,
    /// Physical neighbours.
    prev_phys: u32,
    next_phys: u32,
    /// Free-list neighbours (only meaningful when `free`).
    prev_free: u32,
    next_free: u32,
}

impl BlockNode {
    const fn empty() -> Self {
        Self {
            offset: 0,
            size: 0,
            free: false,
            prev_phys: NIL,
            next_phys: NIL,
            prev_free: NIL,
            next_free: NIL,
        }
    }
}

/// A TLSF allocator covering one contiguous virtual region of bytes
/// `[0, capacity)`. Used to sub-allocate from a single `VkDeviceMemory`.
#[allow(dead_code)]
pub(crate) struct Tlsf {
    capacity: u64,
    /// Block storage. Indices in this vector double as `BlockId`s.
    blocks: Vec<BlockNode>,
    /// Free indices, used as a tiny slab allocator so freeing a block
    /// doesn't grow the storage.
    free_indices: Vec<u32>,
    /// First-level free-list bitmap: bit `i` is set iff some `(i, *)` bin
    /// has at least one free block.
    fl_bitmap: u32,
    /// Second-level free-list bitmaps, one `u32` per FL class. Bit `j` is
    /// set iff bin `(i, j)` has at least one free block.
    sl_bitmap: [u32; FL_INDEX_COUNT as usize],
    /// Per-bin head pointers (`NIL` = empty bin).
    free_heads: Vec<u32>, // FL_INDEX_COUNT * SL_INDEX_COUNT
    /// Statistics.
    used_bytes: u64,
    free_bytes: u64,
    allocation_count: u32,
    free_count: u32,
}

/// Result of a successful TLSF allocation: the byte offset within the
/// parent block, and the side-table id used to free it later.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TlsfAllocation {
    pub offset: u64,
    pub size: u64,
    pub block_id: u32,
}

impl Tlsf {
    pub(crate) fn new(capacity: u64) -> Self {
        let mut s = Self {
            capacity,
            blocks: Vec::with_capacity(64),
            free_indices: Vec::new(),
            fl_bitmap: 0,
            sl_bitmap: [0; FL_INDEX_COUNT as usize],
            free_heads: vec![NIL; (FL_INDEX_COUNT * SL_INDEX_COUNT) as usize],
            used_bytes: 0,
            free_bytes: 0,
            allocation_count: 0,
            free_count: 0,
        };
        // Insert one big free block covering everything.
        let id = s.alloc_node(BlockNode {
            offset: 0,
            size: capacity,
            free: false,
            prev_phys: NIL,
            next_phys: NIL,
            prev_free: NIL,
            next_free: NIL,
        });
        s.free_count = 1;
        s.free_bytes = capacity;
        s.insert_free_block(id);
        s
    }

    pub(crate) fn capacity(&self) -> u64 {
        self.capacity
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Bytes not currently allocated.
    ///
    /// Used only by this module's unit tests, so the non-test `lib` target
    /// correctly sees it as dead. The suppression is scoped to this one method
    /// rather than to the whole `impl`, which is how it was written before: a
    /// block-level allow silences every method in the block, so nobody can see
    /// which one it is actually for. Removing it and reading the compiler is
    /// the only way to find out, and the answer was this method alone.
    #[allow(dead_code)]
    pub(crate) fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    /// Number of currently outstanding allocations.
    pub(crate) fn allocation_count(&self) -> u32 {
        self.allocation_count
    }

    /// Number of free regions (a measure of fragmentation).
    pub(crate) fn free_region_count(&self) -> u32 {
        self.free_count
    }

    /// Try to allocate `size` bytes with `alignment`. Returns `None` if
    /// no free region can satisfy the request.
    pub(crate) fn allocate(&mut self, size: u64, alignment: u64) -> Option<TlsfAllocation> {
        let alignment = max(1, alignment);
        let want = max(SMALL_BLOCK_SIZE, round_up(size, alignment));

        // Walk the free-list bins starting at `mapping(want)`, looking
        // for the first free block whose start (after alignment) plus
        // `want` fits inside the block. This is O(bins * chain_length)
        // worst-case but in practice the first candidate fits.
        let (start_fl, start_sl) = Self::mapping(want);

        for fl in start_fl..FL_INDEX_COUNT {
            let sl_start = if fl == start_fl { start_sl } else { 0 };
            let sl_map = self.sl_bitmap[fl as usize] & (!0u32 << sl_start);
            let mut remaining = sl_map;
            while remaining != 0 {
                let sl = remaining.trailing_zeros();
                remaining &= remaining - 1;
                let bin = Self::bin_index(fl, sl);
                let mut id = self.free_heads[bin];
                while id != NIL {
                    let block = self.blocks[id as usize];
                    let aligned = round_up(block.offset, alignment);
                    let head_padding = aligned - block.offset;
                    if head_padding + want <= block.size {
                        return Some(self.commit_alloc(id, aligned, want));
                    }
                    id = block.next_free;
                }
            }
        }
        None
    }

    /// Convert the chosen free block `id` (with aligned start `aligned`
    /// and size `used_size`) into one used block plus optional
    /// head/tail free splits. Removes the original free block from the
    /// free list and returns the new used block's metadata.
    fn commit_alloc(&mut self, id: u32, aligned: u64, used_size: u64) -> TlsfAllocation {
        let block = self.blocks[id as usize];
        let head_padding = aligned - block.offset;
        let tail_offset = aligned + used_size;
        let tail_padding = block.offset + block.size - tail_offset;

        // Remove the chosen block from the free list and adjust totals.
        self.remove_free_block(id);
        self.free_count -= 1;
        self.free_bytes -= block.size;

        let mut prev_phys = block.prev_phys;
        if head_padding > 0 {
            let head_id = self.alloc_node(BlockNode {
                offset: block.offset,
                size: head_padding,
                free: false,
                prev_phys,
                next_phys: NIL, // patched below
                prev_free: NIL,
                next_free: NIL,
            });
            if prev_phys != NIL {
                self.blocks[prev_phys as usize].next_phys = head_id;
            }
            self.insert_free_block(head_id);
            self.free_count += 1;
            self.free_bytes += head_padding;
            prev_phys = head_id;
        }

        let used_id = self.alloc_node(BlockNode {
            offset: aligned,
            size: used_size,
            free: false,
            prev_phys,
            next_phys: NIL, // patched below
            prev_free: NIL,
            next_free: NIL,
        });
        if prev_phys != NIL {
            self.blocks[prev_phys as usize].next_phys = used_id;
        }

        let next_phys = block.next_phys;
        if tail_padding > 0 {
            let tail_id = self.alloc_node(BlockNode {
                offset: tail_offset,
                size: tail_padding,
                free: false,
                prev_phys: used_id,
                next_phys,
                prev_free: NIL,
                next_free: NIL,
            });
            self.blocks[used_id as usize].next_phys = tail_id;
            if next_phys != NIL {
                self.blocks[next_phys as usize].prev_phys = tail_id;
            }
            self.insert_free_block(tail_id);
            self.free_count += 1;
            self.free_bytes += tail_padding;
        } else {
            self.blocks[used_id as usize].next_phys = next_phys;
            if next_phys != NIL {
                self.blocks[next_phys as usize].prev_phys = used_id;
            }
        }

        self.allocation_count += 1;
        self.used_bytes += used_size;
        // Recycle the original block id.
        self.recycle_node(id);

        TlsfAllocation {
            offset: aligned,
            size: used_size,
            block_id: used_id,
        }
    }

    /// Free a previously allocated block.
    pub(crate) fn free(&mut self, allocation: TlsfAllocation) {
        let mut id = allocation.block_id;
        debug_assert!(!self.blocks[id as usize].free);
        debug_assert_eq!(self.blocks[id as usize].offset, allocation.offset);

        // Move the bytes from used to free. The size on the block stays
        // unchanged at this point (coalescing only redistributes ranges
        // among existing free byte counts).
        let size = self.blocks[id as usize].size;
        self.allocation_count -= 1;
        self.used_bytes -= size;
        self.free_bytes += size;
        self.free_count += 1;

        // Coalesce with previous free physical neighbour.
        let prev = self.blocks[id as usize].prev_phys;
        if prev != NIL && self.blocks[prev as usize].free {
            self.remove_free_block(prev);
            // Merge `id` into `prev`. Bytes are conserved (both sides
            // were already counted in `free_bytes`); only the region
            // count goes down.
            let prev_size = self.blocks[prev as usize].size;
            let id_size = self.blocks[id as usize].size;
            let id_next = self.blocks[id as usize].next_phys;
            self.blocks[prev as usize].size = prev_size + id_size;
            self.blocks[prev as usize].next_phys = id_next;
            if id_next != NIL {
                self.blocks[id_next as usize].prev_phys = prev;
            }
            self.recycle_node(id);
            self.free_count -= 1;
            id = prev;
        }

        // Coalesce with next free physical neighbour.
        let next = self.blocks[id as usize].next_phys;
        if next != NIL && self.blocks[next as usize].free {
            self.remove_free_block(next);
            let id_size = self.blocks[id as usize].size;
            let next_size = self.blocks[next as usize].size;
            let next_next = self.blocks[next as usize].next_phys;
            self.blocks[id as usize].size = id_size + next_size;
            self.blocks[id as usize].next_phys = next_next;
            if next_next != NIL {
                self.blocks[next_next as usize].prev_phys = id;
            }
            self.recycle_node(next);
            self.free_count -= 1;
        }

        self.insert_free_block(id);
    }

    // ----- internal helpers -----

    fn alloc_node(&mut self, node: BlockNode) -> u32 {
        if let Some(idx) = self.free_indices.pop() {
            self.blocks[idx as usize] = node;
            idx
        } else {
            let idx = self.blocks.len() as u32;
            self.blocks.push(node);
            idx
        }
    }

    fn recycle_node(&mut self, id: u32) {
        // We just leave the slot in `blocks` and push the index for reuse.
        self.blocks[id as usize] = BlockNode::empty();
        self.free_indices.push(id);
    }

    fn mapping(size: u64) -> (u32, u32) {
        if size < SMALL_BLOCK_SIZE {
            return (0, 0);
        }
        // First-level: highest set bit.
        let fl_raw = 63u32 - size.leading_zeros();
        let fl = fl_raw - FL_INDEX_SHIFT;
        // Second-level: top SL_INDEX_BITS bits below the leading bit.
        let sl = ((size >> (fl_raw - SL_INDEX_BITS)) ^ (1 << SL_INDEX_BITS)) as u32;
        let fl = fl.min(FL_INDEX_COUNT - 1);
        let sl = sl.min(SL_INDEX_COUNT - 1);
        (fl, sl)
    }

    fn bin_index(fl: u32, sl: u32) -> usize {
        (fl * SL_INDEX_COUNT + sl) as usize
    }

    fn insert_free_block(&mut self, id: u32) {
        let size = self.blocks[id as usize].size;
        let (fl, sl) = Self::mapping(size);
        let bin = Self::bin_index(fl, sl);
        let head = self.free_heads[bin];
        self.blocks[id as usize].free = true;
        self.blocks[id as usize].prev_free = NIL;
        self.blocks[id as usize].next_free = head;
        if head != NIL {
            self.blocks[head as usize].prev_free = id;
        }
        self.free_heads[bin] = id;
        self.fl_bitmap |= 1 << fl;
        self.sl_bitmap[fl as usize] |= 1 << sl;
    }

    fn remove_free_block(&mut self, id: u32) {
        let size = self.blocks[id as usize].size;
        let (fl, sl) = Self::mapping(size);
        let bin = Self::bin_index(fl, sl);
        let prev = self.blocks[id as usize].prev_free;
        let next = self.blocks[id as usize].next_free;
        if prev != NIL {
            self.blocks[prev as usize].next_free = next;
        }
        if next != NIL {
            self.blocks[next as usize].prev_free = prev;
        }
        if self.free_heads[bin] == id {
            self.free_heads[bin] = next;
            if next == NIL {
                self.sl_bitmap[fl as usize] &= !(1 << sl);
                if self.sl_bitmap[fl as usize] == 0 {
                    self.fl_bitmap &= !(1 << fl);
                }
            }
        }
        self.blocks[id as usize].free = false;
        self.blocks[id as usize].prev_free = NIL;
        self.blocks[id as usize].next_free = NIL;
    }
}

fn round_up(value: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (value + mask) & !mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_one_then_free() {
        let mut t = Tlsf::new(1 << 20);
        let a = t.allocate(1024, 256).unwrap();
        assert_eq!(a.size, 1024);
        assert_eq!(a.offset % 256, 0);
        assert_eq!(t.allocation_count(), 1);
        t.free(a);
        assert_eq!(t.allocation_count(), 0);
        // After a free, all bytes should be back in one free region (one
        // big coalesced block).
        assert_eq!(t.free_region_count(), 1);
        assert_eq!(t.free_bytes(), 1 << 20);
    }

    #[test]
    fn allocate_many_distinct_sizes() {
        let mut t = Tlsf::new(1 << 24);
        let mut allocations = Vec::new();
        for i in 0..50 {
            let size = 256 * (i + 1);
            allocations.push(t.allocate(size, 256).unwrap());
        }
        assert_eq!(t.allocation_count(), 50);
        for a in allocations.drain(..) {
            t.free(a);
        }
        assert_eq!(t.allocation_count(), 0);
        // Eventually coalesces back to one block.
        assert_eq!(t.free_region_count(), 1);
    }

    #[test]
    fn allocations_dont_overlap_each_other_or_extend_capacity() {
        let mut t = Tlsf::new(1 << 16);
        let a = t.allocate(4096, 256).unwrap();
        let b = t.allocate(8192, 256).unwrap();
        let c = t.allocate(2048, 256).unwrap();
        // No overlap.
        assert!(disjoint(a.offset, a.size, b.offset, b.size));
        assert!(disjoint(a.offset, a.size, c.offset, c.size));
        assert!(disjoint(b.offset, b.size, c.offset, c.size));
        // All within capacity.
        for x in [a, b, c] {
            assert!(x.offset + x.size <= t.capacity());
        }
    }

    #[test]
    fn alignment_is_respected() {
        let mut t = Tlsf::new(1 << 16);
        for align in [256u64, 512, 1024, 2048, 4096] {
            let a = t.allocate(1024, align).unwrap();
            assert_eq!(a.offset % align, 0, "offset not aligned to {align}");
            t.free(a);
        }
    }

    #[test]
    fn returns_none_when_full() {
        let mut t = Tlsf::new(1 << 12); // 4 KiB
        // Largest single alloc the bin layout allows is the whole capacity.
        let a = t.allocate(4096, 256).unwrap();
        // No more room.
        assert!(t.allocate(256, 256).is_none());
        t.free(a);
        // Free again.
        assert!(t.allocate(256, 256).is_some());
    }

    fn disjoint(a_off: u64, a_sz: u64, b_off: u64, b_sz: u64) -> bool {
        a_off + a_sz <= b_off || b_off + b_sz <= a_off
    }
}
