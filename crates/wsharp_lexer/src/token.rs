//! Token definitions for the W# programming language.

use crate::Span;
use std::fmt;

/// A token produced by the lexer.
#[derive(Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Returns true if this token is trivia (whitespace, comments).
    pub fn is_trivia(&self) -> bool {
        matches!(self.kind, TokenKind::Whitespace | TokenKind::Comment)
    }
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} @ {:?}", self.kind, self.span)
    }
}

/// The kind of token.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenKind {
    // Literals
    /// Integer literal (e.g., 42, 0xFF, 0b1010)
    IntLiteral(i128),
    /// Float literal (e.g., 3.14, 1e-5)
    FloatLiteral(f64),
    /// String literal (e.g., "hello")
    StringLiteral(String),
    /// Character literal (e.g., 'a')
    CharLiteral(char),
    /// Boolean literal
    BoolLiteral(bool),

    // Identifiers
    /// An identifier (e.g., foo, Bar, _test)
    Ident(String),

    // Keywords
    /// `let`
    Let,
    /// `mut`
    Mut,
    /// `fn`
    Fn,
    /// `async`
    Async,
    /// `await`
    Await,
    /// `return`
    Return,
    /// `if`
    If,
    /// `else`
    Else,
    /// `while`
    While,
    /// `for`
    For,
    /// `in`
    In,
    /// `loop`
    Loop,
    /// `break`
    Break,
    /// `continue`
    Continue,
    /// `match`
    Match,
    /// `struct`
    Struct,
    /// `impl`
    Impl,
    /// `trait`
    Trait,
    /// `type`
    Type,
    /// `self`
    SelfLower,
    /// `Self`
    SelfUpper,
    /// `pub`
    Pub,
    /// `mod`
    Mod,
    /// `use`
    Use,
    /// `as`
    As,
    /// `true`
    True,
    /// `false`
    False,
    /// `null`
    Null,
    /// `proto`
    Proto,
    /// `extend`
    Extend,
    /// `new`
    New,

    // HTTP status keywords (for first-class HTTP types)
    /// `http` keyword for HTTP status types
    Http,

    // Operators
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `=`
    Eq,
    /// `==`
    EqEq,
    /// `!=`
    NotEq,
    /// `<`
    Lt,
    /// `<=`
    LtEq,
    /// `>`
    Gt,
    /// `>=`
    GtEq,
    /// `&&`
    AndAnd,
    /// `||`
    OrOr,
    /// `!`
    Not,
    /// `&`
    And,
    /// `|`
    Or,
    /// `^`
    Caret,
    /// `~`
    Tilde,
    /// `<<`
    Shl,
    /// `>>`
    Shr,
    /// `+=`
    PlusEq,
    /// `-=`
    MinusEq,
    /// `*=`
    StarEq,
    /// `/=`
    SlashEq,
    /// `%=`
    PercentEq,
    /// `&=`
    AndEq,
    /// `|=`
    OrEq,
    /// `^=`
    CaretEq,
    /// `<<=`
    ShlEq,
    /// `>>=`
    ShrEq,
    /// `->`
    Arrow,
    /// `=>`
    FatArrow,
    /// `::`
    ColonColon,
    /// `..`
    DotDot,
    /// `...`
    DotDotDot,
    /// `?`
    Question,
    /// `@`
    At,

    // Delimiters
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,

    // Punctuation
    /// `,`
    Comma,
    /// `.`
    Dot,
    /// `:`
    Colon,
    /// `;`
    Semi,

    // Trivia
    /// Whitespace (spaces, tabs, newlines)
    Whitespace,
    /// Comment (single-line or multi-line)
    Comment,

    // Special
    /// End of file
    Eof,
    /// Unknown/invalid character
    Error(char),
}

impl TokenKind {
    /// Returns the keyword for a given identifier, or None if it's not a keyword.
    pub fn keyword(ident: &str) -> Option<TokenKind> {
        Some(match ident {
            "let" => TokenKind::Let,
            "mut" => TokenKind::Mut,
            "fn" => TokenKind::Fn,
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,
            "return" => TokenKind::Return,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "loop" => TokenKind::Loop,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "match" => TokenKind::Match,
            "struct" => TokenKind::Struct,
            "impl" => TokenKind::Impl,
            "trait" => TokenKind::Trait,
            "type" => TokenKind::Type,
            "self" => TokenKind::SelfLower,
            "Self" => TokenKind::SelfUpper,
            "pub" => TokenKind::Pub,
            "mod" => TokenKind::Mod,
            "use" => TokenKind::Use,
            "as" => TokenKind::As,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "proto" => TokenKind::Proto,
            "extend" => TokenKind::Extend,
            "new" => TokenKind::New,
            "http" => TokenKind::Http,
            _ => return None,
        })
    }

    /// Returns the string representation of this token kind.
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenKind::IntLiteral(_) => "integer literal",
            TokenKind::FloatLiteral(_) => "float literal",
            TokenKind::StringLiteral(_) => "string literal",
            TokenKind::CharLiteral(_) => "character literal",
            TokenKind::BoolLiteral(_) => "boolean literal",
            TokenKind::Ident(_) => "identifier",
            TokenKind::Let => "let",
            TokenKind::Mut => "mut",
            TokenKind::Fn => "fn",
            TokenKind::Async => "async",
            TokenKind::Await => "await",
            TokenKind::Return => "return",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::While => "while",
            TokenKind::For => "for",
            TokenKind::In => "in",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Match => "match",
            TokenKind::Struct => "struct",
            TokenKind::Impl => "impl",
            TokenKind::Trait => "trait",
            TokenKind::Type => "type",
            TokenKind::SelfLower => "self",
            TokenKind::SelfUpper => "Self",
            TokenKind::Pub => "pub",
            TokenKind::Mod => "mod",
            TokenKind::Use => "use",
            TokenKind::As => "as",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::Null => "null",
            TokenKind::Proto => "proto",
            TokenKind::Extend => "extend",
            TokenKind::New => "new",
            TokenKind::Http => "http",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Eq => "=",
            TokenKind::EqEq => "==",
            TokenKind::NotEq => "!=",
            TokenKind::Lt => "<",
            TokenKind::LtEq => "<=",
            TokenKind::Gt => ">",
            TokenKind::GtEq => ">=",
            TokenKind::AndAnd => "&&",
            TokenKind::OrOr => "||",
            TokenKind::Not => "!",
            TokenKind::And => "&",
            TokenKind::Or => "|",
            TokenKind::Caret => "^",
            TokenKind::Tilde => "~",
            TokenKind::Shl => "<<",
            TokenKind::Shr => ">>",
            TokenKind::PlusEq => "+=",
            TokenKind::MinusEq => "-=",
            TokenKind::StarEq => "*=",
            TokenKind::SlashEq => "/=",
            TokenKind::PercentEq => "%=",
            TokenKind::AndEq => "&=",
            TokenKind::OrEq => "|=",
            TokenKind::CaretEq => "^=",
            TokenKind::ShlEq => "<<=",
            TokenKind::ShrEq => ">>=",
            TokenKind::Arrow => "->",
            TokenKind::FatArrow => "=>",
            TokenKind::ColonColon => "::",
            TokenKind::DotDot => "..",
            TokenKind::DotDotDot => "...",
            TokenKind::Question => "?",
            TokenKind::At => "@",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Dot => ".",
            TokenKind::Colon => ":",
            TokenKind::Semi => ";",
            TokenKind::Whitespace => "whitespace",
            TokenKind::Comment => "comment",
            TokenKind::Eof => "end of file",
            TokenKind::Error(_) => "error",
        }
    }
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::IntLiteral(n) => write!(f, "{}", n),
            TokenKind::FloatLiteral(n) => write!(f, "{}", n),
            TokenKind::StringLiteral(s) => write!(f, "\"{}\"", s),
            TokenKind::CharLiteral(c) => write!(f, "'{}'", c),
            TokenKind::BoolLiteral(b) => write!(f, "{}", b),
            TokenKind::Ident(s) => write!(f, "{}", s),
            TokenKind::Error(c) => write!(f, "error({})", c),
            _ => write!(f, "{}", self.as_str()),
        }
    }
}
