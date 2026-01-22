//! The main parser implementation.

use crate::error::{ParseError, ParseResult};
use wsharp_ast::*;
use wsharp_lexer::{Lexer, Span, Token, TokenKind};

/// The W# parser.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    source: &'a str,
    next_node_id: u32,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given source code.
    pub fn new(source: &'a str) -> Self {
        let tokens: Vec<Token> = Lexer::new(source)
            .tokenize()
            .into_iter()
            .filter(|t| !t.is_trivia())
            .collect();

        Self {
            tokens,
            pos: 0,
            source,
            next_node_id: 0,
        }
    }

    /// Parses a complete source file.
    pub fn parse(&mut self) -> ParseResult<SourceFile> {
        let start = self.current_span();
        let mut items = Vec::new();

        while !self.is_at_end() {
            items.push(self.parse_item()?);
        }

        let end = if items.is_empty() {
            start
        } else {
            items.last().unwrap().span()
        };

        Ok(SourceFile {
            items,
            span: start.merge(end),
        })
    }

    /// Parses a top-level item.
    pub fn parse_item(&mut self) -> ParseResult<Item> {
        let visibility = self.parse_visibility();

        match self.peek_kind() {
            TokenKind::Fn => Ok(Item::Function(self.parse_function(visibility)?)),
            TokenKind::Async => Ok(Item::Function(self.parse_function(visibility)?)),
            TokenKind::Proto => Ok(Item::Prototype(self.parse_prototype(visibility)?)),
            TokenKind::Extend => Ok(Item::Extension(self.parse_extension()?)),
            TokenKind::Type => Ok(Item::TypeAlias(self.parse_type_alias(visibility)?)),
            TokenKind::Mod => Ok(Item::Module(self.parse_module(visibility)?)),
            TokenKind::Use => Ok(Item::Import(self.parse_import()?)),
            _ => Err(ParseError::UnexpectedToken {
                expected: "fn, proto, extend, type, mod, or use".to_string(),
                found: format!("{}", self.peek_kind()),
                span: self.current_span(),
            }),
        }
    }

    // ========== Helper methods ==========

    pub(crate) fn next_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        NodeId(id)
    }

    pub(crate) fn is_at_end(&self) -> bool {
        self.peek_kind() == TokenKind::Eof
    }

    pub(crate) fn peek(&self) -> Token {
        self.tokens.get(self.pos).cloned().unwrap_or(Token {
            kind: TokenKind::Eof,
            span: Span::dummy(),
        })
    }

    pub(crate) fn peek_kind(&self) -> TokenKind {
        self.peek().kind.clone()
    }

    pub(crate) fn peek_nth(&self, n: usize) -> TokenKind {
        self.tokens
            .get(self.pos + n)
            .map(|t| t.kind.clone())
            .unwrap_or(TokenKind::Eof)
    }

    pub(crate) fn current_span(&self) -> Span {
        self.peek().span
    }

    pub(crate) fn advance(&mut self) -> Token {
        let token = self.peek();
        if !self.is_at_end() {
            self.pos += 1;
        }
        token
    }

    pub(crate) fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.peek_kind()) == std::mem::discriminant(kind)
    }

    pub(crate) fn match_token(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub(crate) fn expect(&mut self, kind: TokenKind) -> ParseResult<Token> {
        if self.check(&kind) {
            Ok(self.advance())
        } else {
            Err(ParseError::unexpected_token(
                kind.as_str(),
                &self.peek_kind(),
                self.current_span(),
            ))
        }
    }

    pub(crate) fn expect_ident(&mut self) -> ParseResult<Ident> {
        match self.peek_kind() {
            TokenKind::Ident(name) => {
                let span = self.advance().span;
                Ok(Ident::new(name, span))
            }
            _ => Err(ParseError::ExpectedIdent {
                span: self.current_span(),
            }),
        }
    }

    fn parse_visibility(&mut self) -> Visibility {
        if self.match_token(&TokenKind::Pub) {
            Visibility::Public
        } else {
            Visibility::Private
        }
    }

    // ========== Declaration parsing ==========

    fn parse_function(&mut self, visibility: Visibility) -> ParseResult<FunctionDecl> {
        let start = self.current_span();
        let is_async = self.match_token(&TokenKind::Async);
        self.expect(TokenKind::Fn)?;

        let name = self.expect_ident()?;
        let generics = self.parse_generics()?;

        self.expect(TokenKind::LParen)?;
        let params = self.parse_parameters()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.match_token(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.check(&TokenKind::LBrace) {
            Some(self.parse_block()?)
        } else {
            self.expect(TokenKind::Semi)?;
            None
        };

        let end = self.current_span();
        let id = self.next_id();

        Ok(FunctionDecl {
            name,
            generics,
            params,
            return_type,
            body,
            is_async,
            visibility,
            dispatch_params: Vec::new(),
            span: start.merge(end),
            id,
        })
    }

    fn parse_generics(&mut self) -> ParseResult<Vec<GenericParam>> {
        if !self.match_token(&TokenKind::Lt) {
            return Ok(Vec::new());
        }

        let mut params = Vec::new();
        loop {
            let start = self.current_span();
            let name = self.expect_ident()?;

            let bounds = if self.match_token(&TokenKind::Colon) {
                self.parse_type_bounds()?
            } else {
                Vec::new()
            };

            let default = if self.match_token(&TokenKind::Eq) {
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push(GenericParam {
                name,
                bounds,
                default,
                span: start.merge(self.current_span()),
            });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(TokenKind::Gt)?;
        Ok(params)
    }

    fn parse_type_bounds(&mut self) -> ParseResult<Vec<TypeExpr>> {
        let mut bounds = vec![self.parse_type()?];
        while self.match_token(&TokenKind::Plus) {
            bounds.push(self.parse_type()?);
        }
        Ok(bounds)
    }

    pub(crate) fn parse_parameters(&mut self) -> ParseResult<Vec<Parameter>> {
        let mut params = Vec::new();

        if self.check(&TokenKind::RParen) {
            return Ok(params);
        }

        loop {
            params.push(self.parse_parameter()?);
            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(params)
    }

    fn parse_parameter(&mut self) -> ParseResult<Parameter> {
        let start = self.current_span();
        // Accept both regular identifiers and 'self' keyword as parameter names
        let name = match self.peek_kind() {
            TokenKind::Ident(name) => {
                let span = self.advance().span;
                Ident::new(name, span)
            }
            TokenKind::SelfLower => {
                let span = self.advance().span;
                Ident::new("self".to_string(), span)
            }
            _ => return Err(ParseError::ExpectedIdent {
                span: self.current_span(),
            }),
        };

        let ty = if self.match_token(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let default = if self.match_token(&TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(Parameter {
            name,
            ty,
            default,
            span: start.merge(self.current_span()),
        })
    }

    fn parse_prototype(&mut self, visibility: Visibility) -> ParseResult<PrototypeDecl> {
        let start = self.current_span();
        self.expect(TokenKind::Proto)?;

        let name = self.expect_ident()?;

        let parent = if self.match_token(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;

        let mut members = Vec::new();
        let mut constructor = None;

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            if self.check(&TokenKind::Fn) || self.check(&TokenKind::Async) {
                let func = self.parse_function(Visibility::Public)?;
                if func.name.name == "new" || func.name.name == "constructor" {
                    constructor = Some(func);
                } else {
                    members.push(PrototypeMember::Method(func));
                }
            } else {
                members.push(self.parse_prototype_property()?);
            }
        }

        self.expect(TokenKind::RBrace)?;

        let id = self.next_id();

        Ok(PrototypeDecl {
            name,
            parent,
            members,
            constructor,
            visibility,
            span: start.merge(self.current_span()),
            id,
        })
    }

    fn parse_prototype_property(&mut self) -> ParseResult<PrototypeMember> {
        let start = self.current_span();
        let name = self.expect_ident()?;

        let ty = if self.match_token(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let default = if self.match_token(&TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.match_token(&TokenKind::Semi);
        self.match_token(&TokenKind::Comma);

        Ok(PrototypeMember::Property {
            name,
            ty,
            default,
            span: start.merge(self.current_span()),
        })
    }

    fn parse_extension(&mut self) -> ParseResult<ExtensionDecl> {
        let start = self.current_span();
        self.expect(TokenKind::Extend)?;

        let target_type = self.parse_type()?;

        self.expect(TokenKind::LBrace)?;

        let mut methods = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            methods.push(self.parse_function(Visibility::Public)?);
        }

        self.expect(TokenKind::RBrace)?;

        let id = self.next_id();

        Ok(ExtensionDecl {
            target_type,
            methods,
            span: start.merge(self.current_span()),
            id,
        })
    }

    fn parse_type_alias(&mut self, visibility: Visibility) -> ParseResult<TypeAliasDecl> {
        let start = self.current_span();
        self.expect(TokenKind::Type)?;

        let name = self.expect_ident()?;
        let generics = self.parse_generics()?;

        self.expect(TokenKind::Eq)?;
        let ty = self.parse_type()?;
        self.expect(TokenKind::Semi)?;

        let id = self.next_id();

        Ok(TypeAliasDecl {
            name,
            generics,
            ty,
            visibility,
            span: start.merge(self.current_span()),
            id,
        })
    }

    fn parse_module(&mut self, visibility: Visibility) -> ParseResult<ModuleDecl> {
        let start = self.current_span();
        self.expect(TokenKind::Mod)?;

        let name = self.expect_ident()?;

        self.expect(TokenKind::LBrace)?;

        let mut items = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            items.push(self.parse_item()?);
        }

        self.expect(TokenKind::RBrace)?;

        let id = self.next_id();

        Ok(ModuleDecl {
            name,
            items,
            visibility,
            span: start.merge(self.current_span()),
            id,
        })
    }

    fn parse_import(&mut self) -> ParseResult<ImportDecl> {
        let start = self.current_span();
        self.expect(TokenKind::Use)?;

        let mut path = vec![self.expect_ident()?];
        while self.match_token(&TokenKind::ColonColon) {
            if self.check(&TokenKind::Star) {
                self.advance();
                self.expect(TokenKind::Semi)?;
                let id = self.next_id();
                return Ok(ImportDecl {
                    path,
                    alias: None,
                    items: ImportItems::All,
                    span: start.merge(self.current_span()),
                    id,
                });
            } else if self.check(&TokenKind::LBrace) {
                self.advance();
                let items = self.parse_import_items()?;
                self.expect(TokenKind::RBrace)?;
                self.expect(TokenKind::Semi)?;
                let id = self.next_id();
                return Ok(ImportDecl {
                    path,
                    alias: None,
                    items: ImportItems::Specific(items),
                    span: start.merge(self.current_span()),
                    id,
                });
            } else {
                path.push(self.expect_ident()?);
            }
        }

        let alias = if self.match_token(&TokenKind::As) {
            Some(self.expect_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::Semi)?;

        let id = self.next_id();

        Ok(ImportDecl {
            path,
            alias,
            items: ImportItems::Module,
            span: start.merge(self.current_span()),
            id,
        })
    }

    fn parse_import_items(&mut self) -> ParseResult<Vec<ImportItem>> {
        let mut items = Vec::new();

        loop {
            let name = self.expect_ident()?;
            let alias = if self.match_token(&TokenKind::As) {
                Some(self.expect_ident()?)
            } else {
                None
            };
            items.push(ImportItem { name, alias });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(items)
    }

    // ========== Type parsing ==========

    pub fn parse_type(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();

        // Check for function type
        if self.check(&TokenKind::Fn) || self.check(&TokenKind::Async) {
            return self.parse_function_type();
        }

        // Parse primary type
        let mut ty = self.parse_primary_type()?;

        // Check for union types (A | B)
        if self.check(&TokenKind::Or) {
            let mut types = vec![ty];
            while self.match_token(&TokenKind::Or) {
                types.push(self.parse_primary_type()?);
            }
            ty = TypeExpr::new(
                TypeExprKind::Union(types),
                start.merge(self.current_span()),
            );
        }

        // Check for optional type (T?)
        if self.match_token(&TokenKind::Question) {
            ty = TypeExpr::new(
                TypeExprKind::Optional(Box::new(ty)),
                start.merge(self.current_span()),
            );
        }

        Ok(ty)
    }

    fn parse_primary_type(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();

        match self.peek_kind() {
            // Named type (check for _ first to handle infer type)
            TokenKind::Ident(ref s) if s == "_" => {
                self.advance();
                Ok(TypeExpr::new(
                    TypeExprKind::Infer,
                    start.merge(self.current_span()),
                ))
            }

            // Named type
            TokenKind::Ident(_) => {
                let name = self.expect_ident()?;

                // Check for generic arguments
                let generics = if self.check(&TokenKind::Lt) {
                    self.parse_type_arguments()?
                } else {
                    Vec::new()
                };

                Ok(TypeExpr::new(
                    TypeExprKind::Named { name, generics },
                    start.merge(self.current_span()),
                ))
            }

            // HTTP status type
            TokenKind::Http => {
                self.advance();
                let http_type = self.parse_http_status_type()?;
                Ok(TypeExpr::new(
                    TypeExprKind::HttpStatus(http_type),
                    start.merge(self.current_span()),
                ))
            }

            // Array type [T] or [T; N]
            TokenKind::LBracket => {
                self.advance();
                let element = self.parse_type()?;
                let size = if self.match_token(&TokenKind::Semi) {
                    match self.peek_kind() {
                        TokenKind::IntLiteral(n) => {
                            self.advance();
                            Some(n as usize)
                        }
                        _ => return Err(ParseError::Expected {
                            expected: "array size",
                            span: self.current_span(),
                        }),
                    }
                } else {
                    None
                };
                self.expect(TokenKind::RBracket)?;
                Ok(TypeExpr::new(
                    TypeExprKind::Array {
                        element: Box::new(element),
                        size,
                    },
                    start.merge(self.current_span()),
                ))
            }

            // Tuple type (A, B, C) or unit ()
            TokenKind::LParen => {
                self.advance();
                if self.match_token(&TokenKind::RParen) {
                    return Ok(TypeExpr::new(
                        TypeExprKind::Unit,
                        start.merge(self.current_span()),
                    ));
                }

                let mut types = vec![self.parse_type()?];
                while self.match_token(&TokenKind::Comma) {
                    types.push(self.parse_type()?);
                }
                self.expect(TokenKind::RParen)?;

                if types.len() == 1 {
                    // Just a parenthesized type
                    Ok(types.pop().unwrap())
                } else {
                    Ok(TypeExpr::new(
                        TypeExprKind::Tuple(types),
                        start.merge(self.current_span()),
                    ))
                }
            }

            // Reference type &T or &mut T
            TokenKind::And => {
                self.advance();
                let mutable = self.match_token(&TokenKind::Mut);
                let inner = self.parse_primary_type()?;
                Ok(TypeExpr::new(
                    TypeExprKind::Ref {
                        inner: Box::new(inner),
                        mutable,
                    },
                    start.merge(self.current_span()),
                ))
            }

            // Never type !
            TokenKind::Not => {
                self.advance();
                Ok(TypeExpr::new(
                    TypeExprKind::Never,
                    start.merge(self.current_span()),
                ))
            }

            _ => Err(ParseError::ExpectedType {
                span: self.current_span(),
            }),
        }
    }

    fn parse_type_arguments(&mut self) -> ParseResult<Vec<TypeExpr>> {
        self.expect(TokenKind::Lt)?;
        let mut args = vec![self.parse_type()?];
        while self.match_token(&TokenKind::Comma) {
            args.push(self.parse_type()?);
        }
        self.expect(TokenKind::Gt)?;
        Ok(args)
    }

    fn parse_function_type(&mut self) -> ParseResult<TypeExpr> {
        let start = self.current_span();
        let is_async = self.match_token(&TokenKind::Async);
        self.expect(TokenKind::Fn)?;

        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        if !self.check(&TokenKind::RParen) {
            params.push(self.parse_type()?);
            while self.match_token(&TokenKind::Comma) {
                params.push(self.parse_type()?);
            }
        }
        self.expect(TokenKind::RParen)?;

        self.expect(TokenKind::Arrow)?;
        let return_type = self.parse_type()?;

        Ok(TypeExpr::new(
            TypeExprKind::Function {
                params,
                return_type: Box::new(return_type),
                is_async,
            },
            start.merge(self.current_span()),
        ))
    }

    fn parse_http_status_type(&mut self) -> ParseResult<HttpStatusTypeExpr> {
        match self.peek_kind() {
            TokenKind::IntLiteral(code) => {
                self.advance();
                let code = code as u16;
                if !(100..=599).contains(&code) {
                    return Err(ParseError::InvalidHttpStatus {
                        code,
                        span: self.current_span(),
                    });
                }
                Ok(HttpStatusTypeExpr::Exact(code))
            }
            TokenKind::Ident(ref s) => {
                let category = match s.as_str() {
                    "1xx" => HttpStatusCategory::Informational,
                    "2xx" => HttpStatusCategory::Success,
                    "3xx" => HttpStatusCategory::Redirection,
                    "4xx" => HttpStatusCategory::ClientError,
                    "5xx" => HttpStatusCategory::ServerError,
                    _ => {
                        return Err(ParseError::Custom {
                            message: format!("invalid HTTP status category: {}", s),
                            span: self.current_span(),
                        })
                    }
                };
                self.advance();
                Ok(HttpStatusTypeExpr::Category(category))
            }
            _ => Err(ParseError::Expected {
                expected: "HTTP status code or category",
                span: self.current_span(),
            }),
        }
    }

    // ========== Block parsing ==========

    pub fn parse_block(&mut self) -> ParseResult<Block> {
        let start = self.current_span();
        self.expect(TokenKind::LBrace)?;

        let mut stmts = Vec::new();
        let mut expr = None;

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let stmt = self.parse_stmt()?;

            // Check if this is an expression without semicolon at the end
            if self.check(&TokenKind::RBrace) {
                if let StmtKind::Expr(e) = stmt.kind {
                    expr = Some(Box::new(e));
                    break;
                }
            }

            stmts.push(stmt);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block {
            stmts,
            expr,
            span: start.merge(self.current_span()),
        })
    }
}
