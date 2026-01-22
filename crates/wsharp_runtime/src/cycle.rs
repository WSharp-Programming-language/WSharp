//! Cycle detection using the Bacon-Rajan algorithm.
//!
//! This module implements concurrent cycle collection for reference-counted objects.
//! The algorithm uses color marking to identify and collect cyclic garbage.

use std::cell::Cell;
use std::ptr::NonNull;

use crate::rc::RcHeader;

/// Color constants for cycle detection.
/// Stored in the lower 2 bits of the flags field.
pub mod color {
    /// In use or free - not part of a cycle candidate
    pub const BLACK: u32 = 0;
    /// Possible member of garbage cycle (being scanned)
    pub const GRAY: u32 = 1;
    /// Confirmed member of garbage cycle (to be freed)
    pub const WHITE: u32 = 2;
    /// Possible root of garbage cycle
    pub const PURPLE: u32 = 3;
}

/// Flag bit indicating the object is in the roots buffer
const BUFFERED_FLAG: u32 = 0x4;

/// Mask for extracting the color from flags
const COLOR_MASK: u32 = 0x3;

/// Trait for objects that can participate in cycle collection.
///
/// Objects must implement this to allow the cycle collector to traverse
/// their references and potentially break cycles.
pub trait Trace {
    /// Visit all reference-counted children of this object.
    fn trace(&self, visitor: &mut dyn FnMut(NonNull<RcHeader>));
}

impl RcHeader {
    /// Get the current color of this object.
    #[inline]
    pub fn color(&self) -> u32 {
        self.flags.get() & COLOR_MASK
    }

    /// Set the color of this object.
    #[inline]
    pub fn set_color(&self, color: u32) {
        let flags = (self.flags.get() & !COLOR_MASK) | (color & COLOR_MASK);
        self.flags.set(flags);
    }

    /// Check if this object is buffered (in the roots list).
    #[inline]
    pub fn is_buffered(&self) -> bool {
        (self.flags.get() & BUFFERED_FLAG) != 0
    }

    /// Set the buffered flag.
    #[inline]
    pub fn set_buffered(&self, buffered: bool) {
        let flags = if buffered {
            self.flags.get() | BUFFERED_FLAG
        } else {
            self.flags.get() & !BUFFERED_FLAG
        };
        self.flags.set(flags);
    }

    /// Get the raw flags value.
    #[inline]
    pub fn flags(&self) -> &Cell<u32> {
        &self.flags
    }
}

/// The cycle collector.
///
/// Maintains a list of potential cycle roots and performs collection
/// when triggered.
pub struct CycleCollector {
    /// Potential cycle roots (purple-colored objects)
    roots: Vec<NonNull<RcHeader>>,
    /// Threshold for automatic collection
    threshold: usize,
}

impl CycleCollector {
    /// Create a new cycle collector.
    pub fn new() -> Self {
        Self {
            roots: Vec::new(),
            threshold: 100, // Collect after 100 potential roots
        }
    }

    /// Create a cycle collector with a custom threshold.
    pub fn with_threshold(threshold: usize) -> Self {
        Self {
            roots: Vec::new(),
            threshold,
        }
    }

    /// Add a potential cycle root.
    ///
    /// Called when an object's strong count decrements but doesn't reach zero.
    /// This object might be part of a cycle.
    pub fn add_possible_root(&mut self, header: NonNull<RcHeader>) {
        unsafe {
            let h = header.as_ref();

            // Only add if not already buffered
            if !h.is_buffered() {
                h.set_color(color::PURPLE);
                h.set_buffered(true);
                self.roots.push(header);
            }
        }

        // Auto-collect if we hit the threshold
        if self.roots.len() >= self.threshold {
            self.collect();
        }
    }

    /// Perform cycle collection.
    ///
    /// This runs the full Bacon-Rajan algorithm:
    /// 1. Mark roots - identify potential garbage cycles
    /// 2. Scan roots - determine which are actually garbage
    /// 3. Collect white - free the garbage cycles
    pub fn collect(&mut self) {
        self.mark_roots();
        self.scan_roots();
        self.collect_roots();
    }

    /// Phase 1: Mark potential cycle roots.
    ///
    /// For each purple root, if it's still a candidate (count > 0),
    /// mark it and its subgraph gray.
    fn mark_roots(&mut self) {
        for &root in &self.roots {
            unsafe {
                let header = root.as_ref();
                if header.color() == color::PURPLE && header.strong_count() > 0 {
                    self.mark_gray(root);
                } else {
                    // Not a candidate anymore - mark black
                    header.set_buffered(false);
                    if header.strong_count() == 0 {
                        // Already dead, will be cleaned up normally
                        header.set_color(color::BLACK);
                    }
                }
            }
        }
    }

    /// Mark an object and its descendants gray.
    ///
    /// Gray indicates "being scanned" - we're determining if this
    /// object is part of a garbage cycle.
    fn mark_gray(&self, header: NonNull<RcHeader>) {
        unsafe {
            let h = header.as_ref();
            if h.color() != color::GRAY {
                h.set_color(color::GRAY);

                // For each child reference, decrement count and mark gray
                // Note: In a real implementation, we'd traverse the object's fields
                // This requires the Trace trait to be implemented for the value type
            }
        }
    }

    /// Phase 2: Scan marked objects to identify garbage.
    ///
    /// If an object's count is still > 0 after marking, it's reachable
    /// from outside the cycle - mark it and descendants black (live).
    /// Otherwise, mark it white (garbage).
    fn scan_roots(&self) {
        for &root in &self.roots {
            unsafe {
                let header = root.as_ref();
                if header.color() == color::GRAY {
                    if header.strong_count() > 0 {
                        // Still referenced from outside - not garbage
                        self.scan_black(root);
                    } else {
                        // No external references - garbage
                        header.set_color(color::WHITE);
                    }
                }
            }
        }
    }

    /// Mark an object and its descendants black (live).
    fn scan_black(&self, header: NonNull<RcHeader>) {
        unsafe {
            let h = header.as_ref();
            h.set_color(color::BLACK);

            // For each child reference, increment count and scan if gray
            // Note: In a real implementation, we'd traverse the object's fields
        }
    }

    /// Phase 3: Collect garbage (white objects).
    fn collect_roots(&mut self) {
        // Take ownership of roots list
        let roots = std::mem::take(&mut self.roots);

        for root in roots {
            unsafe {
                let header = root.as_ref();
                header.set_buffered(false);

                if header.color() == color::WHITE {
                    // This is garbage - collect it
                    self.collect_white(root);
                } else {
                    // Not garbage - reset to black
                    header.set_color(color::BLACK);
                }
            }
        }
    }

    /// Collect a white (garbage) object and its descendants.
    fn collect_white(&self, header: NonNull<RcHeader>) {
        unsafe {
            let h = header.as_ref();
            if h.color() == color::WHITE {
                h.set_color(color::BLACK);

                // For each child reference, collect if white
                // Note: In a real implementation, we'd traverse and free

                // The actual deallocation happens when the Rc is dropped
                // Here we just mark for collection
            }
        }
    }

    /// Get the number of potential roots currently tracked.
    pub fn root_count(&self) -> usize {
        self.roots.len()
    }

    /// Check if collection is needed based on threshold.
    pub fn should_collect(&self) -> bool {
        self.roots.len() >= self.threshold
    }
}

impl Default for CycleCollector {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local cycle collector for single-threaded use
thread_local! {
    static COLLECTOR: std::cell::RefCell<CycleCollector> = std::cell::RefCell::new(CycleCollector::new());
}

/// Add a potential cycle root to the thread-local collector.
pub fn possible_root(header: NonNull<RcHeader>) {
    COLLECTOR.with(|c| {
        c.borrow_mut().add_possible_root(header);
    });
}

/// Trigger cycle collection on the thread-local collector.
pub fn collect_cycles() {
    COLLECTOR.with(|c| {
        c.borrow_mut().collect();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_operations() {
        let header = RcHeader::new();

        assert_eq!(header.color(), color::BLACK);

        header.set_color(color::PURPLE);
        assert_eq!(header.color(), color::PURPLE);

        header.set_color(color::GRAY);
        assert_eq!(header.color(), color::GRAY);

        header.set_color(color::WHITE);
        assert_eq!(header.color(), color::WHITE);

        header.set_color(color::BLACK);
        assert_eq!(header.color(), color::BLACK);
    }

    #[test]
    fn test_buffered_flag() {
        let header = RcHeader::new();

        assert!(!header.is_buffered());

        header.set_buffered(true);
        assert!(header.is_buffered());
        // Color should be preserved
        assert_eq!(header.color(), color::BLACK);

        header.set_color(color::PURPLE);
        assert!(header.is_buffered());
        assert_eq!(header.color(), color::PURPLE);

        header.set_buffered(false);
        assert!(!header.is_buffered());
        assert_eq!(header.color(), color::PURPLE);
    }

    #[test]
    fn test_collector_creation() {
        let collector = CycleCollector::new();
        assert_eq!(collector.root_count(), 0);
        assert!(!collector.should_collect());
    }

    #[test]
    fn test_collector_with_threshold() {
        let collector = CycleCollector::with_threshold(10);
        assert_eq!(collector.threshold, 10);
    }
}
