//! Reference counting implementation for W#.

use std::cell::Cell;
use std::ptr::NonNull;
use std::marker::PhantomData;
use std::ops::Deref;

/// The header for reference-counted objects.
#[repr(C)]
pub struct RcHeader {
    /// Strong reference count
    strong_count: Cell<u32>,
    /// Weak reference count
    weak_count: Cell<u32>,
    /// Flags for cycle detection and other metadata
    pub(crate) flags: Cell<u32>,
}

impl RcHeader {
    pub fn new() -> Self {
        Self {
            strong_count: Cell::new(1),
            weak_count: Cell::new(0),
            flags: Cell::new(0),
        }
    }

    pub fn strong_count(&self) -> u32 {
        self.strong_count.get()
    }

    pub fn weak_count(&self) -> u32 {
        self.weak_count.get()
    }

    pub fn increment_strong(&self) {
        let count = self.strong_count.get();
        self.strong_count.set(count.saturating_add(1));
    }

    pub fn decrement_strong(&self) -> u32 {
        let count = self.strong_count.get();
        let new_count = count.saturating_sub(1);
        self.strong_count.set(new_count);
        new_count
    }

    pub fn increment_weak(&self) {
        let count = self.weak_count.get();
        self.weak_count.set(count.saturating_add(1));
    }

    pub fn decrement_weak(&self) -> u32 {
        let count = self.weak_count.get();
        let new_count = count.saturating_sub(1);
        self.weak_count.set(new_count);
        new_count
    }
}

impl Default for RcHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// The inner structure of a reference-counted value.
#[repr(C)]
struct RcInner<T> {
    header: RcHeader,
    value: T,
}

/// A reference-counted smart pointer.
pub struct Rc<T> {
    ptr: NonNull<RcInner<T>>,
    _marker: PhantomData<RcInner<T>>,
}

impl<T> Rc<T> {
    /// Creates a new reference-counted value.
    pub fn new(value: T) -> Self {
        let inner = Box::new(RcInner {
            header: RcHeader::new(),
            value,
        });
        Self {
            ptr: NonNull::new(Box::into_raw(inner)).unwrap(),
            _marker: PhantomData,
        }
    }

    /// Returns the strong reference count.
    pub fn strong_count(&self) -> u32 {
        unsafe { self.ptr.as_ref().header.strong_count() }
    }

    /// Returns the weak reference count.
    pub fn weak_count(&self) -> u32 {
        unsafe { self.ptr.as_ref().header.weak_count() }
    }

    /// Creates a weak reference to this value.
    pub fn downgrade(&self) -> Weak<T> {
        unsafe {
            self.ptr.as_ref().header.increment_weak();
        }
        Weak {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable reference if this is the only strong reference.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.strong_count() == 1 && self.weak_count() == 0 {
            Some(unsafe { &mut self.ptr.as_mut().value })
        } else {
            None
        }
    }

    /// Returns a pointer to the header for cycle detection.
    ///
    /// # Safety
    /// The returned pointer is valid as long as the Rc exists.
    pub(crate) fn header_ptr(&self) -> NonNull<RcHeader> {
        // Safety: The inner struct starts with the header, so we can cast
        unsafe {
            NonNull::new_unchecked(self.ptr.as_ptr() as *mut RcHeader)
        }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(&self) -> Self {
        unsafe {
            self.ptr.as_ref().header.increment_strong();
        }
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(&mut self) {
        unsafe {
            let count = self.ptr.as_ref().header.decrement_strong();
            if count == 0 {
                // Drop the value
                std::ptr::drop_in_place(&mut self.ptr.as_mut().value);

                // If no weak references, deallocate
                if self.ptr.as_ref().header.weak_count() == 0 {
                    drop(Box::from_raw(self.ptr.as_ptr()));
                }
            } else {
                // Count > 0: This might be part of a cycle.
                // Add to the cycle collector's potential roots.
                crate::cycle::possible_root(self.header_ptr());
            }
        }
    }
}

impl<T> Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &self.ptr.as_ref().value }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Rc<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

/// A weak reference to a reference-counted value.
pub struct Weak<T> {
    ptr: NonNull<RcInner<T>>,
    _marker: PhantomData<RcInner<T>>,
}

impl<T> Weak<T> {
    /// Attempts to upgrade to a strong reference.
    pub fn upgrade(&self) -> Option<Rc<T>> {
        unsafe {
            let count = self.ptr.as_ref().header.strong_count();
            if count == 0 {
                None
            } else {
                self.ptr.as_ref().header.increment_strong();
                Some(Rc {
                    ptr: self.ptr,
                    _marker: PhantomData,
                })
            }
        }
    }

    /// Returns the strong reference count.
    pub fn strong_count(&self) -> u32 {
        unsafe { self.ptr.as_ref().header.strong_count() }
    }
}

impl<T> Clone for Weak<T> {
    fn clone(&self) -> Self {
        unsafe {
            self.ptr.as_ref().header.increment_weak();
        }
        Self {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }
}

impl<T> Drop for Weak<T> {
    fn drop(&mut self) {
        unsafe {
            let weak_count = self.ptr.as_ref().header.decrement_weak();
            let strong_count = self.ptr.as_ref().header.strong_count();
            if weak_count == 0 && strong_count == 0 {
                drop(Box::from_raw(self.ptr.as_ptr()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rc_basic() {
        let rc = Rc::new(42);
        assert_eq!(*rc, 42);
        assert_eq!(rc.strong_count(), 1);
    }

    #[test]
    fn test_rc_clone() {
        let rc1 = Rc::new(42);
        let rc2 = rc1.clone();
        assert_eq!(*rc1, 42);
        assert_eq!(*rc2, 42);
        assert_eq!(rc1.strong_count(), 2);
    }

    #[test]
    fn test_weak_reference() {
        let rc = Rc::new(42);
        let weak = rc.downgrade();
        assert_eq!(rc.weak_count(), 1);
        assert!(weak.upgrade().is_some());

        drop(rc);
        assert!(weak.upgrade().is_none());
    }
}
