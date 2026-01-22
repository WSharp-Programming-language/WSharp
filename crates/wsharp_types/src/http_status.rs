//! HTTP status code type utilities.

use crate::types::{HttpStatusType, HttpStatusTypeKind, StatusCategory};

/// Runtime representation of an HTTP status code.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HttpStatusValue {
    pub code: u16,
    pub category: u8,
}

impl HttpStatusValue {
    pub fn new(code: u16) -> Self {
        Self {
            code,
            category: match code {
                100..=199 => 1,
                200..=299 => 2,
                300..=399 => 3,
                400..=499 => 4,
                500..=599 => 5,
                _ => 0,
            },
        }
    }

    pub fn is_informational(&self) -> bool {
        self.category == 1
    }

    pub fn is_success(&self) -> bool {
        self.category == 2
    }

    pub fn is_redirection(&self) -> bool {
        self.category == 3
    }

    pub fn is_client_error(&self) -> bool {
        self.category == 4
    }

    pub fn is_server_error(&self) -> bool {
        self.category == 5
    }

    pub fn is_error(&self) -> bool {
        self.category >= 4
    }
}

/// Type checking utilities for HTTP status types.
pub fn check_http_status_assignment(
    expected: &HttpStatusType,
    actual: &HttpStatusType,
) -> Result<(), HttpStatusTypeError> {
    match (&expected.kind, &actual.kind) {
        // Exact to exact: must match
        (HttpStatusTypeKind::Exact(e), HttpStatusTypeKind::Exact(a)) => {
            if e == a {
                Ok(())
            } else {
                Err(HttpStatusTypeError::Mismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
        }

        // Category to exact: actual must be in category
        (HttpStatusTypeKind::Category(cat), HttpStatusTypeKind::Exact(code)) => {
            if cat.contains(*code) {
                Ok(())
            } else {
                Err(HttpStatusTypeError::Mismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
        }

        // Exact to category: expected code must be in category
        (HttpStatusTypeKind::Exact(code), HttpStatusTypeKind::Category(cat)) => {
            if cat.contains(*code) {
                Ok(())
            } else {
                Err(HttpStatusTypeError::Mismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
        }

        // Range to exact: actual must be in range
        (HttpStatusTypeKind::Range { start, end }, HttpStatusTypeKind::Exact(code)) => {
            if code >= start && code <= end {
                Ok(())
            } else {
                Err(HttpStatusTypeError::Mismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                })
            }
        }

        // Any matches anything
        (HttpStatusTypeKind::Any, _) | (_, HttpStatusTypeKind::Any) => Ok(()),

        // Other combinations need more complex checking
        _ => Ok(()), // TODO: Implement full type compatibility checking
    }
}

/// Error in HTTP status type checking.
#[derive(Clone, Debug)]
pub enum HttpStatusTypeError {
    Mismatch {
        expected: HttpStatusType,
        actual: HttpStatusType,
    },
    InvalidStatusCode(u16),
}

/// Parse an HTTP status pattern from source (e.g., "2xx", "404", "4xx | 5xx")
pub fn parse_http_status_pattern(s: &str) -> Option<HttpStatusTypeKind> {
    let s = s.trim();

    // Check for exact code
    if let Ok(code) = s.parse::<u16>() {
        return Some(HttpStatusTypeKind::Exact(code));
    }

    // Check for category patterns (1xx, 2xx, etc.)
    if s.len() == 3 && s.ends_with("xx") {
        match s.chars().next()? {
            '1' => return Some(HttpStatusTypeKind::Category(StatusCategory::Informational)),
            '2' => return Some(HttpStatusTypeKind::Category(StatusCategory::Success)),
            '3' => return Some(HttpStatusTypeKind::Category(StatusCategory::Redirection)),
            '4' => return Some(HttpStatusTypeKind::Category(StatusCategory::ClientError)),
            '5' => return Some(HttpStatusTypeKind::Category(StatusCategory::ServerError)),
            _ => return None,
        }
    }

    // Check for range (e.g., "400..499")
    if let Some((start, end)) = s.split_once("..") {
        let start = start.trim().parse::<u16>().ok()?;
        let end = end.trim().parse::<u16>().ok()?;
        return Some(HttpStatusTypeKind::Range { start, end });
    }

    // Check for union (e.g., "4xx | 5xx")
    if s.contains('|') {
        let parts: Vec<HttpStatusTypeKind> = s
            .split('|')
            .filter_map(|part| parse_http_status_pattern(part.trim()))
            .collect();
        if !parts.is_empty() {
            return Some(HttpStatusTypeKind::Union(parts));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_status_value() {
        let ok = HttpStatusValue::new(200);
        assert!(ok.is_success());
        assert!(!ok.is_error());

        let not_found = HttpStatusValue::new(404);
        assert!(not_found.is_client_error());
        assert!(not_found.is_error());

        let internal_error = HttpStatusValue::new(500);
        assert!(internal_error.is_server_error());
        assert!(internal_error.is_error());
    }

    #[test]
    fn test_parse_http_status_pattern() {
        assert_eq!(
            parse_http_status_pattern("200"),
            Some(HttpStatusTypeKind::Exact(200))
        );

        assert_eq!(
            parse_http_status_pattern("2xx"),
            Some(HttpStatusTypeKind::Category(StatusCategory::Success))
        );

        assert_eq!(
            parse_http_status_pattern("400..499"),
            Some(HttpStatusTypeKind::Range { start: 400, end: 499 })
        );

        let union = parse_http_status_pattern("4xx | 5xx");
        assert!(matches!(union, Some(HttpStatusTypeKind::Union(_))));
    }
}
