//! Multiple dispatch resolution for W#.

use crate::types::{Type, HttpStatusTypeKind};
use std::collections::HashMap;

/// A unique identifier for a function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionId(pub u32);

/// A dispatch table for multiple dispatch resolution.
#[derive(Clone, Debug, Default)]
pub struct DispatchTable {
    entries: HashMap<String, Vec<DispatchEntry>>,
}

/// An entry in the dispatch table.
#[derive(Clone, Debug)]
pub struct DispatchEntry {
    pub signature: Vec<TypePattern>,
    pub implementation: FunctionId,
    pub specificity: u32,
}

/// A pattern for type matching in dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum TypePattern {
    /// Matches an exact type
    Exact(Type),
    /// Matches a type and all subtypes
    Subtype(Type),
    /// Matches HTTP status codes
    HttpStatus(HttpStatusPattern),
    /// Matches a specific function type
    FunctionSignature(Type),
    /// Matches any type
    Any,
}

/// HTTP status pattern for dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum HttpStatusPattern {
    /// Exact status code
    Exact(u16),
    /// Range of status codes
    Range { start: u16, end: u16 },
    /// Category of status codes
    Category(crate::types::StatusCategory),
    /// Any status code
    Any,
}

/// Error during dispatch resolution.
#[derive(Clone, Debug, PartialEq)]
pub enum DispatchError {
    /// No applicable method found
    NoApplicableMethod {
        function_name: String,
        arg_types: Vec<Type>,
    },
    /// Multiple methods with same specificity
    AmbiguousDispatch {
        function_name: String,
        candidates: Vec<FunctionId>,
    },
}

impl DispatchTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a dispatch entry for a function.
    pub fn register(&mut self, name: String, entry: DispatchEntry) {
        self.entries.entry(name).or_default().push(entry);
    }

    /// Returns all entries for a function name.
    pub fn entries_for(&self, name: &str) -> impl Iterator<Item = &DispatchEntry> {
        self.entries.get(name).into_iter().flatten()
    }

    /// Resolves the most specific applicable method for the given arguments.
    pub fn resolve(
        &self,
        function_name: &str,
        arg_types: &[Type],
    ) -> Result<FunctionId, DispatchError> {
        let applicable: Vec<_> = self
            .entries_for(function_name)
            .filter(|entry| Self::is_applicable(entry, arg_types))
            .collect();

        if applicable.is_empty() {
            return Err(DispatchError::NoApplicableMethod {
                function_name: function_name.to_string(),
                arg_types: arg_types.to_vec(),
            });
        }

        // Sort by specificity (highest first)
        let mut sorted = applicable;
        sorted.sort_by_key(|e| std::cmp::Reverse(e.specificity));

        // Check for ambiguity
        if sorted.len() > 1 && sorted[0].specificity == sorted[1].specificity {
            return Err(DispatchError::AmbiguousDispatch {
                function_name: function_name.to_string(),
                candidates: sorted
                    .iter()
                    .take_while(|e| e.specificity == sorted[0].specificity)
                    .map(|e| e.implementation)
                    .collect(),
            });
        }

        Ok(sorted[0].implementation)
    }

    /// Checks if a dispatch entry is applicable to the given argument types.
    fn is_applicable(entry: &DispatchEntry, arg_types: &[Type]) -> bool {
        if entry.signature.len() != arg_types.len() {
            return false;
        }

        entry
            .signature
            .iter()
            .zip(arg_types.iter())
            .all(|(pattern, arg_type)| Self::matches_pattern(pattern, arg_type))
    }

    /// Checks if a type matches a pattern.
    fn matches_pattern(pattern: &TypePattern, ty: &Type) -> bool {
        match pattern {
            TypePattern::Any => true,
            TypePattern::Exact(expected) => ty == expected,
            TypePattern::Subtype(base) => Self::is_subtype(ty, base),
            TypePattern::HttpStatus(pattern) => Self::matches_http_status(pattern, ty),
            TypePattern::FunctionSignature(expected) => ty == expected,
        }
    }

    /// Checks if `sub` is a subtype of `super_`.
    fn is_subtype(sub: &Type, super_: &Type) -> bool {
        // Basic subtype check - can be extended for prototype inheritance
        sub == super_
    }

    /// Checks if a type matches an HTTP status pattern.
    fn matches_http_status(pattern: &HttpStatusPattern, ty: &Type) -> bool {
        let Type::HttpStatus(status) = ty else {
            return false;
        };

        match (&status.kind, pattern) {
            (HttpStatusTypeKind::Exact(code), HttpStatusPattern::Exact(expected)) => {
                code == expected
            }
            (HttpStatusTypeKind::Exact(code), HttpStatusPattern::Range { start, end }) => {
                code >= start && code <= end
            }
            (HttpStatusTypeKind::Exact(code), HttpStatusPattern::Category(cat)) => {
                cat.contains(*code)
            }
            (_, HttpStatusPattern::Any) => true,
            _ => false,
        }
    }
}

/// Calculates the specificity of a type pattern.
pub fn calculate_specificity(patterns: &[TypePattern]) -> u32 {
    patterns
        .iter()
        .map(|p| match p {
            TypePattern::Any => 0,
            TypePattern::Subtype(_) => 1,
            TypePattern::HttpStatus(HttpStatusPattern::Any) => 1,
            TypePattern::HttpStatus(HttpStatusPattern::Category(_)) => 2,
            TypePattern::HttpStatus(HttpStatusPattern::Range { .. }) => 3,
            TypePattern::HttpStatus(HttpStatusPattern::Exact(_)) => 4,
            TypePattern::Exact(_) => 4,
            TypePattern::FunctionSignature(_) => 4,
        })
        .sum()
}
