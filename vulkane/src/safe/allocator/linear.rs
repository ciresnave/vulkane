//! A simple linear bump allocator over a single device-memory block.
//!
//! Useful for stack-like / ring-buffer / single-frame upload patterns where
//! the entire pool is reset all at once. Allocation and free are both O(1)
//! and there is no fragmentation, but individual allocations cannot be freed
//! independently — only [`Linear::reset`] returns memory to the pool.
//!
//! For general-purpose long-lived allocations, use the
//! [TLSF allocator](super::tlsf::Tlsf) instead.

use std::cmp::max;

#[derive(Debug, Clone, Copy)]
pub(crate) struct LinearAllocation {
    pub offset: u64,
    pub size: u64,
}

pub(crate) struct Linear {
    capacity: u64,
    cursor: u64,
    allocation_count: u32,
}

impl Linear {
    pub(crate) fn new(capacity: u64) -> Self {
        Self {
            capacity,
            cursor: 0,
            allocation_count: 0,
        }
    }

    pub(crate) fn used_bytes(&self) -> u64 {
        self.cursor
    }

    pub(crate) fn free_bytes(&self) -> u64 {
        self.capacity - self.cursor
    }

    pub(crate) fn allocation_count(&self) -> u32 {
        self.allocation_count
    }

    /// Bump-allocate `size` bytes with `alignment`. Returns `None` if the
    /// remaining space is insufficient.
    pub(crate) fn allocate(&mut self, size: u64, alignment: u64) -> Option<LinearAllocation> {
        let align = max(1, alignment);
        let aligned = (self.cursor + align - 1) & !(align - 1);
        let end = aligned.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        self.cursor = end;
        self.allocation_count += 1;
        Some(LinearAllocation {
            offset: aligned,
            size,
        })
    }

    /// Reset the bump pointer to the start of the block. After this call
    /// every previously allocated `LinearAllocation` is invalid — the
    /// caller is responsible for ensuring no live references remain.
    pub(crate) fn reset(&mut self) {
        self.cursor = 0;
        self.allocation_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_in_order() {
        let mut a = Linear::new(1024);
        let r1 = a.allocate(100, 1).unwrap();
        let r2 = a.allocate(200, 1).unwrap();
        assert_eq!(r1.offset, 0);
        assert_eq!(r2.offset, 100);
    }

    #[test]
    fn alignment_advances_cursor() {
        let mut a = Linear::new(1024);
        let _ = a.allocate(1, 1).unwrap();
        let r = a.allocate(1, 256).unwrap();
        assert_eq!(r.offset, 256);
    }

    #[test]
    fn reset_returns_capacity() {
        let mut a = Linear::new(1024);
        a.allocate(500, 1).unwrap();
        assert_eq!(a.used_bytes(), 500);
        a.reset();
        assert_eq!(a.used_bytes(), 0);
        assert_eq!(a.free_bytes(), 1024);
    }

    #[test]
    fn returns_none_when_full() {
        let mut a = Linear::new(64);
        a.allocate(60, 1).unwrap();
        assert!(a.allocate(8, 1).is_none());
    }
}
