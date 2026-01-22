//! Source location tracking for error messages and debugging.

use std::fmt;

/// A span represents a range of bytes in the source code.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// The starting byte offset (inclusive).
    pub start: u32,
    /// The ending byte offset (exclusive).
    pub end: u32,
}

impl Span {
    /// Creates a new span from start to end.
    pub fn new(start: u32, end: u32) -> Self {
        debug_assert!(start <= end, "Span start must be <= end");
        Self { start, end }
    }

    /// Creates a span covering a single byte position.
    pub fn point(pos: u32) -> Self {
        Self { start: pos, end: pos + 1 }
    }

    /// Returns the length of the span in bytes.
    pub fn len(&self) -> u32 {
        self.end - self.start
    }

    /// Returns true if the span is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Creates a new span that covers both this span and another.
    pub fn merge(&self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Returns the dummy span used for synthetic tokens.
    pub fn dummy() -> Self {
        Self { start: 0, end: 0 }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

/// A value with an associated source span.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} @ {:?}", self.node, self.span)
    }
}
