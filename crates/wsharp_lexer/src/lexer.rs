//! The lexer implementation for W#.

use crate::{Span, Token, TokenKind};
use std::str::Chars;

/// The lexer for W# source code.
pub struct Lexer<'a> {
    /// The source code being lexed.
    source: &'a str,
    /// Iterator over source characters.
    chars: Chars<'a>,
    /// Current byte position in the source.
    pos: u32,
    /// The character at the current position (None if at end).
    current: Option<char>,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source code.
    pub fn new(source: &'a str) -> Self {
        let mut chars = source.chars();
        let current = chars.next();
        Self {
            source,
            chars,
            pos: 0,
            current,
        }
    }

    /// Returns the next token from the source.
    pub fn next_token(&mut self) -> Token {
        self.skip_trivia();

        let start = self.pos;

        let kind = match self.current {
            None => TokenKind::Eof,

            Some(c) if c.is_ascii_alphabetic() || c == '_' => self.lex_ident_or_keyword(),

            Some(c) if c.is_ascii_digit() => self.lex_number(),

            Some('"') => self.lex_string(),

            Some('\'') => self.lex_char(),

            Some(c) => self.lex_punctuation(c),
        };

        let span = Span::new(start, self.pos);
        Token::new(kind, span)
    }

    /// Tokenizes the entire source and returns all tokens.
    pub fn tokenize(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }

    /// Advances to the next character and returns the current one.
    fn advance(&mut self) -> Option<char> {
        let current = self.current;
        if let Some(c) = current {
            self.pos += c.len_utf8() as u32;
            self.current = self.chars.next();
        }
        current
    }

    /// Returns the current character without advancing.
    fn peek(&self) -> Option<char> {
        self.current
    }

    /// Returns the next character without advancing.
    fn peek_next(&self) -> Option<char> {
        self.chars.clone().next()
    }

    /// Returns true if the current character matches the expected one.
    fn check(&self, expected: char) -> bool {
        self.current == Some(expected)
    }

    /// Advances if the current character matches the expected one.
    fn match_char(&mut self, expected: char) -> bool {
        if self.check(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skips whitespace and comments.
    fn skip_trivia(&mut self) {
        loop {
            match self.current {
                Some(' ' | '\t' | '\n' | '\r') => {
                    self.advance();
                }
                Some('/') if self.peek_next() == Some('/') => {
                    // Single-line comment
                    while self.current.is_some() && self.current != Some('\n') {
                        self.advance();
                    }
                }
                Some('/') if self.peek_next() == Some('*') => {
                    // Multi-line comment
                    self.advance(); // /
                    self.advance(); // *
                    let mut depth = 1;
                    while depth > 0 {
                        match (self.current, self.peek_next()) {
                            (Some('*'), Some('/')) => {
                                self.advance();
                                self.advance();
                                depth -= 1;
                            }
                            (Some('/'), Some('*')) => {
                                self.advance();
                                self.advance();
                                depth += 1;
                            }
                            (Some(_), _) => {
                                self.advance();
                            }
                            (None, _) => break, // Unterminated comment
                        }
                    }
                }
                _ => break,
            }
        }
    }

    /// Lexes an identifier or keyword.
    fn lex_ident_or_keyword(&mut self) -> TokenKind {
        let start = self.pos as usize;

        while let Some(c) = self.current {
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end = self.pos as usize;
        let ident = &self.source[start..end];

        // Check for boolean literals first
        if ident == "true" {
            return TokenKind::BoolLiteral(true);
        }
        if ident == "false" {
            return TokenKind::BoolLiteral(false);
        }

        // Check for keywords
        TokenKind::keyword(ident).unwrap_or_else(|| TokenKind::Ident(ident.to_string()))
    }

    /// Lexes a number (integer or float).
    fn lex_number(&mut self) -> TokenKind {
        let start = self.pos as usize;

        // Check for hex, binary, or octal
        if self.check('0') {
            self.advance();
            match self.current {
                Some('x' | 'X') => return self.lex_hex_number(),
                Some('b' | 'B') => return self.lex_binary_number(),
                Some('o' | 'O') => return self.lex_octal_number(),
                _ => {}
            }
        }

        // Decimal number
        while let Some(c) = self.current {
            if c.is_ascii_digit() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        // Check for float
        if self.check('.') && self.peek_next().is_some_and(|c| c.is_ascii_digit()) {
            self.advance(); // .
            while let Some(c) = self.current {
                if c.is_ascii_digit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }

            // Check for exponent
            if self.current == Some('e') || self.current == Some('E') {
                self.advance();
                if self.current == Some('+') || self.current == Some('-') {
                    self.advance();
                }
                while let Some(c) = self.current {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
            }

            let end = self.pos as usize;
            let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
            return TokenKind::FloatLiteral(text.parse().unwrap_or(0.0));
        }

        // Check for exponent without decimal point (e.g., 1e10)
        if self.current == Some('e') || self.current == Some('E') {
            self.advance();
            if self.current == Some('+') || self.current == Some('-') {
                self.advance();
            }
            while let Some(c) = self.current {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            let end = self.pos as usize;
            let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
            return TokenKind::FloatLiteral(text.parse().unwrap_or(0.0));
        }

        let end = self.pos as usize;
        let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
        TokenKind::IntLiteral(text.parse().unwrap_or(0))
    }

    fn lex_hex_number(&mut self) -> TokenKind {
        self.advance(); // x or X
        let start = self.pos as usize;

        while let Some(c) = self.current {
            if c.is_ascii_hexdigit() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end = self.pos as usize;
        let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
        TokenKind::IntLiteral(i128::from_str_radix(&text, 16).unwrap_or(0))
    }

    fn lex_binary_number(&mut self) -> TokenKind {
        self.advance(); // b or B
        let start = self.pos as usize;

        while let Some(c) = self.current {
            if c == '0' || c == '1' || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end = self.pos as usize;
        let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
        TokenKind::IntLiteral(i128::from_str_radix(&text, 2).unwrap_or(0))
    }

    fn lex_octal_number(&mut self) -> TokenKind {
        self.advance(); // o or O
        let start = self.pos as usize;

        while let Some(c) = self.current {
            if ('0'..='7').contains(&c) || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let end = self.pos as usize;
        let text: String = self.source[start..end].chars().filter(|c| *c != '_').collect();
        TokenKind::IntLiteral(i128::from_str_radix(&text, 8).unwrap_or(0))
    }

    /// Lexes a string literal.
    fn lex_string(&mut self) -> TokenKind {
        self.advance(); // Opening "
        let mut value = String::new();

        loop {
            match self.current {
                None | Some('\n') => {
                    // Unterminated string
                    break;
                }
                Some('"') => {
                    self.advance();
                    break;
                }
                Some('\\') => {
                    self.advance();
                    match self.current {
                        Some('n') => {
                            value.push('\n');
                            self.advance();
                        }
                        Some('r') => {
                            value.push('\r');
                            self.advance();
                        }
                        Some('t') => {
                            value.push('\t');
                            self.advance();
                        }
                        Some('\\') => {
                            value.push('\\');
                            self.advance();
                        }
                        Some('"') => {
                            value.push('"');
                            self.advance();
                        }
                        Some('0') => {
                            value.push('\0');
                            self.advance();
                        }
                        Some(c) => {
                            value.push(c);
                            self.advance();
                        }
                        None => {}
                    }
                }
                Some(c) => {
                    value.push(c);
                    self.advance();
                }
            }
        }

        TokenKind::StringLiteral(value)
    }

    /// Lexes a character literal.
    fn lex_char(&mut self) -> TokenKind {
        self.advance(); // Opening '

        let c = match self.current {
            Some('\\') => {
                self.advance();
                match self.current {
                    Some('n') => '\n',
                    Some('r') => '\r',
                    Some('t') => '\t',
                    Some('\\') => '\\',
                    Some('\'') => '\'',
                    Some('0') => '\0',
                    Some(c) => c,
                    None => '\0',
                }
            }
            Some(c) => c,
            None => '\0',
        };

        self.advance(); // The character

        if self.match_char('\'') {
            TokenKind::CharLiteral(c)
        } else {
            // Unterminated char literal, just return what we have
            TokenKind::CharLiteral(c)
        }
    }

    /// Lexes punctuation and operators.
    fn lex_punctuation(&mut self, c: char) -> TokenKind {
        self.advance();

        match c {
            '+' => {
                if self.match_char('=') {
                    TokenKind::PlusEq
                } else {
                    TokenKind::Plus
                }
            }
            '-' => {
                if self.match_char('=') {
                    TokenKind::MinusEq
                } else if self.match_char('>') {
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            '*' => {
                if self.match_char('=') {
                    TokenKind::StarEq
                } else {
                    TokenKind::Star
                }
            }
            '/' => {
                if self.match_char('=') {
                    TokenKind::SlashEq
                } else {
                    TokenKind::Slash
                }
            }
            '%' => {
                if self.match_char('=') {
                    TokenKind::PercentEq
                } else {
                    TokenKind::Percent
                }
            }
            '=' => {
                if self.match_char('=') {
                    TokenKind::EqEq
                } else if self.match_char('>') {
                    TokenKind::FatArrow
                } else {
                    TokenKind::Eq
                }
            }
            '!' => {
                if self.match_char('=') {
                    TokenKind::NotEq
                } else {
                    TokenKind::Not
                }
            }
            '<' => {
                if self.match_char('=') {
                    TokenKind::LtEq
                } else if self.match_char('<') {
                    if self.match_char('=') {
                        TokenKind::ShlEq
                    } else {
                        TokenKind::Shl
                    }
                } else {
                    TokenKind::Lt
                }
            }
            '>' => {
                if self.match_char('=') {
                    TokenKind::GtEq
                } else if self.match_char('>') {
                    if self.match_char('=') {
                        TokenKind::ShrEq
                    } else {
                        TokenKind::Shr
                    }
                } else {
                    TokenKind::Gt
                }
            }
            '&' => {
                if self.match_char('&') {
                    TokenKind::AndAnd
                } else if self.match_char('=') {
                    TokenKind::AndEq
                } else {
                    TokenKind::And
                }
            }
            '|' => {
                if self.match_char('|') {
                    TokenKind::OrOr
                } else if self.match_char('=') {
                    TokenKind::OrEq
                } else {
                    TokenKind::Or
                }
            }
            '^' => {
                if self.match_char('=') {
                    TokenKind::CaretEq
                } else {
                    TokenKind::Caret
                }
            }
            '~' => TokenKind::Tilde,
            '?' => TokenKind::Question,
            '@' => TokenKind::At,
            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            '{' => TokenKind::LBrace,
            '}' => TokenKind::RBrace,
            ',' => TokenKind::Comma,
            '.' => {
                if self.match_char('.') {
                    if self.match_char('.') {
                        TokenKind::DotDotDot
                    } else {
                        TokenKind::DotDot
                    }
                } else {
                    TokenKind::Dot
                }
            }
            ':' => {
                if self.match_char(':') {
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            ';' => TokenKind::Semi,
            _ => TokenKind::Error(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = "let x = 42;";
        let tokens = Lexer::new(source).tokenize();

        assert_eq!(tokens.len(), 6);
        assert!(matches!(tokens[0].kind, TokenKind::Let));
        assert!(matches!(&tokens[1].kind, TokenKind::Ident(s) if s == "x"));
        assert!(matches!(tokens[2].kind, TokenKind::Eq));
        assert!(matches!(tokens[3].kind, TokenKind::IntLiteral(42)));
        assert!(matches!(tokens[4].kind, TokenKind::Semi));
        assert!(matches!(tokens[5].kind, TokenKind::Eof));
    }

    #[test]
    fn test_string_literal() {
        let source = r#""hello world""#;
        let tokens = Lexer::new(source).tokenize();

        assert!(matches!(&tokens[0].kind, TokenKind::StringLiteral(s) if s == "hello world"));
    }

    #[test]
    fn test_operators() {
        let source = "+ - * / == != <= >= && || -> =>";
        let tokens = Lexer::new(source).tokenize();

        let expected = [
            TokenKind::Plus,
            TokenKind::Minus,
            TokenKind::Star,
            TokenKind::Slash,
            TokenKind::EqEq,
            TokenKind::NotEq,
            TokenKind::LtEq,
            TokenKind::GtEq,
            TokenKind::AndAnd,
            TokenKind::OrOr,
            TokenKind::Arrow,
            TokenKind::FatArrow,
            TokenKind::Eof,
        ];

        for (token, expected) in tokens.iter().zip(expected.iter()) {
            assert_eq!(&token.kind, expected);
        }
    }

    #[test]
    fn test_hex_binary_octal() {
        let tokens = Lexer::new("0xFF 0b1010 0o77").tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::IntLiteral(255)));
        assert!(matches!(tokens[1].kind, TokenKind::IntLiteral(10)));
        assert!(matches!(tokens[2].kind, TokenKind::IntLiteral(63)));
    }

    #[test]
    fn test_float() {
        let tokens = Lexer::new("3.14 1e10 2.5e-3").tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::FloatLiteral(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[1].kind, TokenKind::FloatLiteral(f) if (f - 1e10).abs() < 1.0));
        assert!(matches!(tokens[2].kind, TokenKind::FloatLiteral(f) if (f - 0.0025).abs() < 0.0001));
    }

    #[test]
    fn test_comments() {
        let source = "let x = 1; // comment\nlet y = 2;";
        let tokens = Lexer::new(source).tokenize();

        // Comments are skipped
        let idents: Vec<_> = tokens
            .iter()
            .filter_map(|t| {
                if let TokenKind::Ident(s) = &t.kind {
                    Some(s.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(idents, vec!["x", "y"]);
    }

    #[test]
    fn test_keywords() {
        let source = "fn async await if else while for return";
        let tokens = Lexer::new(source).tokenize();

        assert!(matches!(tokens[0].kind, TokenKind::Fn));
        assert!(matches!(tokens[1].kind, TokenKind::Async));
        assert!(matches!(tokens[2].kind, TokenKind::Await));
        assert!(matches!(tokens[3].kind, TokenKind::If));
        assert!(matches!(tokens[4].kind, TokenKind::Else));
        assert!(matches!(tokens[5].kind, TokenKind::While));
        assert!(matches!(tokens[6].kind, TokenKind::For));
        assert!(matches!(tokens[7].kind, TokenKind::Return));
    }
}
