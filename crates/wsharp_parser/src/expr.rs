//! Expression parsing with Pratt parsing for operator precedence.

use crate::error::{ParseError, ParseResult};
use crate::parser::Parser;
use wsharp_ast::*;
use wsharp_lexer::TokenKind;

impl Parser<'_> {
    /// Parses an expression.
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let expr = self.parse_or()?;

        // Check for assignment
        if self.match_token(&TokenKind::Eq) {
            let value = self.parse_assignment()?;
            let id = self.next_id();
            return Ok(Expr::new(
                ExprKind::Assign {
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                start.merge(self.current_span()),
                id,
            ));
        }

        // Check for compound assignment
        if let Some(op) = self.match_compound_assign() {
            let value = self.parse_assignment()?;
            let id = self.next_id();
            return Ok(Expr::new(
                ExprKind::CompoundAssign {
                    op,
                    target: Box::new(expr),
                    value: Box::new(value),
                },
                start.merge(self.current_span()),
                id,
            ));
        }

        Ok(expr)
    }

    fn match_compound_assign(&mut self) -> Option<BinaryOp> {
        let op = match self.peek_kind() {
            TokenKind::PlusEq => BinaryOp::Add,
            TokenKind::MinusEq => BinaryOp::Sub,
            TokenKind::StarEq => BinaryOp::Mul,
            TokenKind::SlashEq => BinaryOp::Div,
            TokenKind::PercentEq => BinaryOp::Mod,
            TokenKind::AndEq => BinaryOp::BitAnd,
            TokenKind::OrEq => BinaryOp::BitOr,
            TokenKind::CaretEq => BinaryOp::BitXor,
            TokenKind::ShlEq => BinaryOp::Shl,
            TokenKind::ShrEq => BinaryOp::Shr,
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    fn parse_or(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_and()?;

        while self.match_token(&TokenKind::OrOr) {
            let right = self.parse_and()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_and(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_bitor()?;

        while self.match_token(&TokenKind::AndAnd) {
            let right = self.parse_bitor()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_bitor(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_bitxor()?;

        while self.match_token(&TokenKind::Or) {
            let right = self.parse_bitxor()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitOr,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_bitxor(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_bitand()?;

        while self.match_token(&TokenKind::Caret) {
            let right = self.parse_bitand()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitXor,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_bitand(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_equality()?;

        while self.match_token(&TokenKind::And) {
            let right = self.parse_equality()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op: BinaryOp::BitAnd,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_equality(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_comparison()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::EqEq => BinaryOp::Eq,
                TokenKind::NotEq => BinaryOp::NotEq,
                _ => break,
            };
            self.advance();

            let right = self.parse_comparison()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_comparison(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_shift()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Lt => BinaryOp::Lt,
                TokenKind::LtEq => BinaryOp::LtEq,
                TokenKind::Gt => BinaryOp::Gt,
                TokenKind::GtEq => BinaryOp::GtEq,
                _ => break,
            };
            self.advance();

            let right = self.parse_shift()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_shift(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_term()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Shl => BinaryOp::Shl,
                TokenKind::Shr => BinaryOp::Shr,
                _ => break,
            };
            self.advance();

            let right = self.parse_term()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_term(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_factor()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub,
                _ => break,
            };
            self.advance();

            let right = self.parse_factor()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_factor(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.peek_kind() {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent => BinaryOp::Mod,
                _ => break,
            };
            self.advance();

            let right = self.parse_unary()?;
            let id = self.next_id();
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                start.merge(self.current_span()),
                id,
            );
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();

        let op = match self.peek_kind() {
            TokenKind::Minus => UnaryOp::Neg,
            TokenKind::Not => UnaryOp::Not,
            TokenKind::Tilde => UnaryOp::BitNot,
            TokenKind::And => {
                self.advance();
                if self.match_token(&TokenKind::Mut) {
                    let operand = self.parse_unary()?;
                    let id = self.next_id();
                    return Ok(Expr::new(
                        ExprKind::Unary {
                            op: UnaryOp::RefMut,
                            operand: Box::new(operand),
                        },
                        start.merge(self.current_span()),
                        id,
                    ));
                } else {
                    let operand = self.parse_unary()?;
                    let id = self.next_id();
                    return Ok(Expr::new(
                        ExprKind::Unary {
                            op: UnaryOp::Ref,
                            operand: Box::new(operand),
                        },
                        start.merge(self.current_span()),
                        id,
                    ));
                }
            }
            TokenKind::Star => UnaryOp::Deref,
            _ => return self.parse_await_expr(),
        };

        self.advance();
        let operand = self.parse_unary()?;
        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::Unary {
                op,
                operand: Box::new(operand),
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_await_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();

        if self.match_token(&TokenKind::Await) {
            let operand = self.parse_postfix()?;
            let id = self.next_id();
            return Ok(Expr::new(
                ExprKind::Await(Box::new(operand)),
                start.merge(self.current_span()),
                id,
            ));
        }

        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let mut expr = self.parse_primary()?;

        loop {
            match self.peek_kind() {
                // Function call
                TokenKind::LParen => {
                    self.advance();
                    let args = self.parse_call_args()?;
                    self.expect(TokenKind::RParen)?;
                    let id = self.next_id();
                    expr = Expr::new(
                        ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        start.merge(self.current_span()),
                        id,
                    );
                }

                // Property/method access
                TokenKind::Dot => {
                    self.advance();
                    let property = self.expect_ident()?;

                    // Check if it's a method call
                    if self.check(&TokenKind::LParen) {
                        self.advance();
                        let args = self.parse_call_args()?;
                        self.expect(TokenKind::RParen)?;
                        let id = self.next_id();
                        expr = Expr::new(
                            ExprKind::MethodCall {
                                receiver: Box::new(expr),
                                method: property,
                                args,
                            },
                            start.merge(self.current_span()),
                            id,
                        );
                    } else {
                        let id = self.next_id();
                        expr = Expr::new(
                            ExprKind::PropertyAccess {
                                object: Box::new(expr),
                                property,
                            },
                            start.merge(self.current_span()),
                            id,
                        );
                    }
                }

                // Index access
                TokenKind::LBracket => {
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    let id = self.next_id();
                    expr = Expr::new(
                        ExprKind::IndexAccess {
                            object: Box::new(expr),
                            index: Box::new(index),
                        },
                        start.merge(self.current_span()),
                        id,
                    );
                }

                // Type cast
                TokenKind::As => {
                    self.advance();
                    let ty = self.parse_type()?;
                    let id = self.next_id();
                    expr = Expr::new(
                        ExprKind::Cast {
                            expr: Box::new(expr),
                            ty,
                        },
                        start.merge(self.current_span()),
                        id,
                    );
                }

                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_call_args(&mut self) -> ParseResult<Vec<Expr>> {
        let mut args = Vec::new();

        if !self.check(&TokenKind::RParen) {
            args.push(self.parse_expr()?);
            while self.match_token(&TokenKind::Comma) {
                args.push(self.parse_expr()?);
            }
        }

        Ok(args)
    }

    fn parse_primary(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();

        match self.peek_kind() {
            // Literals
            TokenKind::IntLiteral(n) => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Int(n)),
                    start,
                    id,
                ))
            }

            TokenKind::FloatLiteral(n) => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Float(n)),
                    start,
                    id,
                ))
            }

            TokenKind::StringLiteral(s) => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::String(s)),
                    start,
                    id,
                ))
            }

            TokenKind::CharLiteral(c) => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Char(c)),
                    start,
                    id,
                ))
            }

            TokenKind::BoolLiteral(b) => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Bool(b)),
                    start,
                    id,
                ))
            }

            TokenKind::Null => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Null),
                    start,
                    id,
                ))
            }

            // Identifier
            TokenKind::Ident(_) => {
                let ident = self.expect_ident()?;
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Ident(ident),
                    start.merge(self.current_span()),
                    id,
                ))
            }

            // Self keyword (as expression)
            TokenKind::SelfLower => {
                let span = self.advance().span;
                let ident = Ident::new("self".to_string(), span);
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Ident(ident),
                    start.merge(span),
                    id,
                ))
            }

            // HTTP status literal
            TokenKind::Http => {
                self.advance();
                match self.peek_kind() {
                    TokenKind::IntLiteral(code) => {
                        self.advance();
                        let id = self.next_id();
                        Ok(Expr::new(
                            ExprKind::HttpStatus(code as u16),
                            start.merge(self.current_span()),
                            id,
                        ))
                    }
                    _ => Err(ParseError::Expected {
                        expected: "HTTP status code",
                        span: self.current_span(),
                    }),
                }
            }

            // Parenthesized expression or tuple
            TokenKind::LParen => {
                self.advance();
                if self.match_token(&TokenKind::RParen) {
                    // Unit value ()
                    let id = self.next_id();
                    return Ok(Expr::new(
                        ExprKind::Tuple(Vec::new()),
                        start.merge(self.current_span()),
                        id,
                    ));
                }

                let first = self.parse_expr()?;

                if self.match_token(&TokenKind::Comma) {
                    // Tuple
                    let mut elements = vec![first];
                    if !self.check(&TokenKind::RParen) {
                        elements.push(self.parse_expr()?);
                        while self.match_token(&TokenKind::Comma) {
                            if self.check(&TokenKind::RParen) {
                                break;
                            }
                            elements.push(self.parse_expr()?);
                        }
                    }
                    self.expect(TokenKind::RParen)?;
                    let id = self.next_id();
                    Ok(Expr::new(
                        ExprKind::Tuple(elements),
                        start.merge(self.current_span()),
                        id,
                    ))
                } else {
                    // Parenthesized expression
                    self.expect(TokenKind::RParen)?;
                    Ok(first)
                }
            }

            // Array literal
            TokenKind::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                if !self.check(&TokenKind::RBracket) {
                    elements.push(self.parse_expr()?);
                    while self.match_token(&TokenKind::Comma) {
                        if self.check(&TokenKind::RBracket) {
                            break;
                        }
                        elements.push(self.parse_expr()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::ArrayLiteral(elements),
                    start.merge(self.current_span()),
                    id,
                ))
            }

            // Object literal
            TokenKind::LBrace => {
                self.parse_object_literal()
            }

            // If expression
            TokenKind::If => {
                self.parse_if_expr()
            }

            // Match expression
            TokenKind::Match => {
                self.parse_match_expr()
            }

            // While loop
            TokenKind::While => {
                self.parse_while_expr()
            }

            // For loop
            TokenKind::For => {
                self.parse_for_expr()
            }

            // Loop
            TokenKind::Loop => {
                self.parse_loop_expr()
            }

            // Break
            TokenKind::Break => {
                self.advance();
                let value = if !self.check(&TokenKind::Semi) && !self.check(&TokenKind::RBrace) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Break(value),
                    start.merge(self.current_span()),
                    id,
                ))
            }

            // Continue
            TokenKind::Continue => {
                self.advance();
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Continue,
                    start.merge(self.current_span()),
                    id,
                ))
            }

            // Return
            TokenKind::Return => {
                self.advance();
                let value = if !self.check(&TokenKind::Semi) && !self.check(&TokenKind::RBrace) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::Return(value),
                    start.merge(self.current_span()),
                    id,
                ))
            }

            // Lambda (fn syntax or closure syntax)
            TokenKind::Fn | TokenKind::Async | TokenKind::Or | TokenKind::OrOr => {
                self.parse_lambda()
            }

            // New expression
            TokenKind::New => {
                self.advance();
                let prototype = self.parse_postfix()?;
                self.expect(TokenKind::LParen)?;
                let args = self.parse_call_args()?;
                self.expect(TokenKind::RParen)?;
                let id = self.next_id();
                Ok(Expr::new(
                    ExprKind::New {
                        prototype: Box::new(prototype),
                        args,
                    },
                    start.merge(self.current_span()),
                    id,
                ))
            }

            _ => Err(ParseError::ExpectedExpression {
                span: self.current_span(),
            }),
        }
    }

    fn parse_object_literal(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::LBrace)?;

        let mut members = Vec::new();

        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            let key = self.expect_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            let member_span = key.span.merge(value.span);
            members.push(ObjectMember {
                key,
                value,
                span: member_span,
            });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(TokenKind::RBrace)?;
        let id = self.next_id();

        Ok(Expr::new(
            ExprKind::ObjectLiteral {
                prototype: None,
                members,
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_if_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::If)?;

        let condition = self.parse_expr()?;
        let then_branch = self.parse_block()?;

        let else_branch = if self.match_token(&TokenKind::Else) {
            if self.check(&TokenKind::If) {
                // else if
                let else_if = self.parse_if_expr()?;
                Some(Box::new(Block {
                    stmts: Vec::new(),
                    expr: Some(Box::new(else_if)),
                    span: self.current_span(),
                }))
            } else {
                Some(Box::new(self.parse_block()?))
            }
        } else {
            None
        };

        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch,
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_match_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::Match)?;

        let scrutinee = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;

        let mut arms = Vec::new();
        while !self.check(&TokenKind::RBrace) && !self.is_at_end() {
            arms.push(self.parse_match_arm()?);
        }

        self.expect(TokenKind::RBrace)?;
        let id = self.next_id();

        Ok(Expr::new(
            ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_match_arm(&mut self) -> ParseResult<MatchArm> {
        let start = self.current_span();
        let pattern = self.parse_pattern()?;

        let guard = if self.match_token(&TokenKind::If) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.expect(TokenKind::FatArrow)?;
        let body = self.parse_expr()?;

        self.match_token(&TokenKind::Comma);

        Ok(MatchArm {
            pattern,
            guard,
            body,
            span: start.merge(self.current_span()),
        })
    }

    fn parse_pattern(&mut self) -> ParseResult<Pattern> {
        let start = self.current_span();

        match self.peek_kind() {
            // Wildcard
            TokenKind::Ident(ref s) if s == "_" => {
                self.advance();
                Ok(Pattern::new(PatternKind::Wildcard, start))
            }

            // Literal patterns
            TokenKind::IntLiteral(n) => {
                self.advance();
                Ok(Pattern::new(
                    PatternKind::Literal(Literal::Int(n)),
                    start,
                ))
            }

            TokenKind::StringLiteral(s) => {
                self.advance();
                Ok(Pattern::new(
                    PatternKind::Literal(Literal::String(s)),
                    start,
                ))
            }

            TokenKind::BoolLiteral(b) => {
                self.advance();
                Ok(Pattern::new(
                    PatternKind::Literal(Literal::Bool(b)),
                    start,
                ))
            }

            // HTTP status pattern
            TokenKind::Http => {
                self.advance();
                match self.peek_kind() {
                    TokenKind::IntLiteral(code) => {
                        self.advance();
                        Ok(Pattern::new(
                            PatternKind::HttpStatus(HttpStatusPattern::Exact(code as u16)),
                            start.merge(self.current_span()),
                        ))
                    }
                    TokenKind::Ident(ref s) => {
                        let category = match s.as_str() {
                            "1xx" => HttpStatusCategory::Informational,
                            "2xx" => HttpStatusCategory::Success,
                            "3xx" => HttpStatusCategory::Redirection,
                            "4xx" => HttpStatusCategory::ClientError,
                            "5xx" => HttpStatusCategory::ServerError,
                            _ => return Err(ParseError::Custom {
                                message: format!("invalid HTTP status category: {}", s),
                                span: self.current_span(),
                            }),
                        };
                        self.advance();
                        Ok(Pattern::new(
                            PatternKind::HttpStatus(HttpStatusPattern::Category(category)),
                            start.merge(self.current_span()),
                        ))
                    }
                    _ => Err(ParseError::Expected {
                        expected: "HTTP status code or category",
                        span: self.current_span(),
                    }),
                }
            }

            // Identifier pattern
            TokenKind::Ident(_) => {
                let mutable = self.match_token(&TokenKind::Mut);
                let name = self.expect_ident()?;

                // Check for @ binding
                if self.match_token(&TokenKind::At) {
                    let inner = self.parse_pattern()?;
                    return Ok(Pattern::new(
                        PatternKind::Binding {
                            name,
                            pattern: Box::new(inner),
                        },
                        start.merge(self.current_span()),
                    ));
                }

                Ok(Pattern::new(
                    PatternKind::Ident { name, mutable },
                    start.merge(self.current_span()),
                ))
            }

            // Tuple pattern
            TokenKind::LParen => {
                self.advance();
                let mut patterns = Vec::new();
                if !self.check(&TokenKind::RParen) {
                    patterns.push(self.parse_pattern()?);
                    while self.match_token(&TokenKind::Comma) {
                        patterns.push(self.parse_pattern()?);
                    }
                }
                self.expect(TokenKind::RParen)?;
                Ok(Pattern::new(
                    PatternKind::Tuple(patterns),
                    start.merge(self.current_span()),
                ))
            }

            // Array pattern
            TokenKind::LBracket => {
                self.advance();
                let mut elements = Vec::new();
                let mut rest = None;

                while !self.check(&TokenKind::RBracket) && !self.is_at_end() {
                    if self.match_token(&TokenKind::DotDot) {
                        if matches!(self.peek_kind(), TokenKind::Ident(_)) {
                            rest = Some(Box::new(self.parse_pattern()?));
                        }
                        break;
                    }
                    elements.push(self.parse_pattern()?);
                    if !self.match_token(&TokenKind::Comma) {
                        break;
                    }
                }

                self.expect(TokenKind::RBracket)?;
                Ok(Pattern::new(
                    PatternKind::Array { elements, rest },
                    start.merge(self.current_span()),
                ))
            }

            _ => Err(ParseError::Expected {
                expected: "pattern",
                span: self.current_span(),
            }),
        }
    }

    fn parse_while_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::While)?;

        let condition = self.parse_expr()?;
        let body = self.parse_block()?;

        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::While {
                condition: Box::new(condition),
                body: Box::new(body),
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_for_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::For)?;

        let binding = self.expect_ident()?;
        self.expect(TokenKind::In)?;
        let iterator = self.parse_expr()?;
        let body = self.parse_block()?;

        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::For {
                binding,
                iterator: Box::new(iterator),
                body: Box::new(body),
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_loop_expr(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        self.expect(TokenKind::Loop)?;

        let body = self.parse_block()?;

        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::Loop {
                body: Box::new(body),
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    fn parse_lambda(&mut self) -> ParseResult<Expr> {
        let start = self.current_span();
        let is_async = self.match_token(&TokenKind::Async);

        // Parse parameters - supports both fn(params) and |params| or || syntax
        let params = if self.match_token(&TokenKind::OrOr) {
            // || - closure with no parameters
            Vec::new()
        } else if self.match_token(&TokenKind::Or) {
            // |params| - closure with parameters between pipes
            let params = self.parse_closure_params()?;
            self.expect(TokenKind::Or)?;
            params
        } else {
            // fn(params) - traditional function syntax
            self.expect(TokenKind::Fn)?;
            self.expect(TokenKind::LParen)?;
            let params = self.parse_parameters()?;
            self.expect(TokenKind::RParen)?;
            params
        };

        let return_type = if self.match_token(&TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.check(&TokenKind::LBrace) {
            let block = self.parse_block()?;
            let block_span = block.span;
            let id = self.next_id();
            Expr::new(ExprKind::Block(Box::new(block)), block_span, id)
        } else {
            self.expect(TokenKind::FatArrow)?;
            self.parse_expr()?
        };

        let id = self.next_id();
        Ok(Expr::new(
            ExprKind::Lambda {
                params,
                return_type,
                body: Box::new(body),
                is_async,
            },
            start.merge(self.current_span()),
            id,
        ))
    }

    /// Parse closure parameters (simpler than function parameters, types are optional)
    fn parse_closure_params(&mut self) -> ParseResult<Vec<Parameter>> {
        let mut params = Vec::new();

        // Check if we're at the closing pipe already (empty params)
        if self.check(&TokenKind::Or) {
            return Ok(params);
        }

        loop {
            let param_start = self.current_span();
            let name = self.expect_ident()?;

            // Type annotation is optional for closure params
            let ty = if self.match_token(&TokenKind::Colon) {
                Some(self.parse_type()?)
            } else {
                None
            };

            params.push(Parameter {
                name,
                ty,
                default: None,
                span: param_start.merge(self.current_span()),
            });

            if !self.match_token(&TokenKind::Comma) {
                break;
            }
        }

        Ok(params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_expr(source: &str) -> Expr {
        let mut parser = Parser::new(source);
        parser.parse_expr().expect("failed to parse expression")
    }

    fn parse_source(source: &str) -> wsharp_ast::SourceFile {
        let mut parser = Parser::new(source);
        parser.parse().expect("failed to parse source file")
    }

    // ========== If Expression Tests ==========

    #[test]
    fn test_simple_if() {
        let source = "if x { 1 }";
        let expr = parse_expr(source);
        assert!(matches!(expr.kind, ExprKind::If { .. }));
    }

    #[test]
    fn test_if_else() {
        let source = "if x { 1 } else { 2 }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::If { else_branch, .. } => {
                assert!(else_branch.is_some());
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn test_if_else_if() {
        let source = "if x { 1 } else if y { 2 } else { 3 }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::If { else_branch, .. } => {
                assert!(else_branch.is_some());
                let else_block = else_branch.unwrap();
                assert!(else_block.expr.is_some());
                let inner = else_block.expr.as_ref().unwrap();
                assert!(matches!(inner.kind, ExprKind::If { .. }));
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn test_if_with_comparison() {
        let source = "if x > 0 { x } else { 0 }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::If { condition, .. } => {
                assert!(matches!(condition.kind, ExprKind::Binary { op: BinaryOp::Gt, .. }));
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn test_if_with_logical_and() {
        let source = "if x > 0 && y < 10 { 1 }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::If { condition, .. } => {
                assert!(matches!(condition.kind, ExprKind::Binary { op: BinaryOp::And, .. }));
            }
            _ => panic!("expected if expression"),
        }
    }

    #[test]
    fn test_if_with_logical_or() {
        let source = "if x || y { 1 }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::If { condition, .. } => {
                assert!(matches!(condition.kind, ExprKind::Binary { op: BinaryOp::Or, .. }));
            }
            _ => panic!("expected if expression"),
        }
    }

    // ========== While Loop Tests ==========

    #[test]
    fn test_simple_while() {
        let source = "while x { y; }";
        let expr = parse_expr(source);
        assert!(matches!(expr.kind, ExprKind::While { .. }));
    }

    #[test]
    fn test_while_with_comparison() {
        let source = "while i < 10 { i = i + 1; }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::While { condition, .. } => {
                assert!(matches!(condition.kind, ExprKind::Binary { op: BinaryOp::Lt, .. }));
            }
            _ => panic!("expected while expression"),
        }
    }

    #[test]
    fn test_while_with_break() {
        let source = "while true { break; }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::While { body, .. } => {
                // break; is parsed as a statement in the block
                // Either stmts is non-empty, or expr contains the break
                assert!(!body.stmts.is_empty() || body.expr.is_some());
            }
            _ => panic!("expected while expression"),
        }
    }

    #[test]
    fn test_while_with_continue() {
        let source = "while x { continue; }";
        let expr = parse_expr(source);
        assert!(matches!(expr.kind, ExprKind::While { .. }));
    }

    // ========== For Loop Tests ==========

    #[test]
    fn test_simple_for() {
        let source = "for i in items { process(i); }";
        let expr = parse_expr(source);
        assert!(matches!(expr.kind, ExprKind::For { .. }));
    }

    #[test]
    fn test_for_with_range() {
        let source = "for i in range { x = x + i; }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::For { binding, .. } => {
                assert_eq!(binding.name, "i");
            }
            _ => panic!("expected for expression"),
        }
    }

    #[test]
    fn test_for_with_array() {
        let source = "for item in [1, 2, 3] { sum = sum + item; }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::For { iterator, .. } => {
                assert!(matches!(iterator.kind, ExprKind::ArrayLiteral(_)));
            }
            _ => panic!("expected for expression"),
        }
    }

    // ========== Loop Expression Tests ==========

    #[test]
    fn test_simple_loop() {
        let source = "loop { x; }";
        let expr = parse_expr(source);
        assert!(matches!(expr.kind, ExprKind::Loop { .. }));
    }

    #[test]
    fn test_loop_with_break_value() {
        let source = "loop { break 42; }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Loop { body } => {
                // break 42; could be in stmts or as the tail expr
                let has_break = if !body.stmts.is_empty() {
                    if let StmtKind::Expr(ref e) = body.stmts[0].kind {
                        matches!(&e.kind, ExprKind::Break(Some(_)))
                    } else {
                        false
                    }
                } else if let Some(ref e) = body.expr {
                    matches!(&e.kind, ExprKind::Break(Some(_)))
                } else {
                    false
                };
                assert!(has_break, "expected break with value in loop body");
            }
            _ => panic!("expected loop expression"),
        }
    }

    #[test]
    fn test_nested_loops() {
        let source = "loop { while x { break; } }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Loop { body } => {
                assert!(!body.stmts.is_empty() || body.expr.is_some());
            }
            _ => panic!("expected loop expression"),
        }
    }

    // ========== Match Expression Tests ==========

    #[test]
    fn test_simple_match() {
        let source = "match x { 1 => one, 2 => two, _ => other }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Match { arms, .. } => {
                assert_eq!(arms.len(), 3);
            }
            _ => panic!("expected match expression"),
        }
    }

    #[test]
    fn test_match_with_guard() {
        let source = "match x { n if n > 0 => positive, _ => other }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Match { arms, .. } => {
                assert!(arms[0].guard.is_some());
            }
            _ => panic!("expected match expression"),
        }
    }

    #[test]
    fn test_match_with_tuple_pattern() {
        let source = "match pair { (a, b) => a, _ => zero }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Match { arms, .. } => {
                assert!(matches!(arms[0].pattern.kind, PatternKind::Tuple(_)));
            }
            _ => panic!("expected match expression"),
        }
    }

    #[test]
    fn test_match_with_wildcard() {
        let source = "match x { _ => default_value }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Match { arms, .. } => {
                assert!(matches!(arms[0].pattern.kind, PatternKind::Wildcard));
            }
            _ => panic!("expected match expression"),
        }
    }

    #[test]
    fn test_match_with_binding() {
        let source = "match x { n @ 42 => n, _ => zero }";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Match { arms, .. } => {
                assert!(matches!(arms[0].pattern.kind, PatternKind::Binding { .. }));
            }
            _ => panic!("expected match expression"),
        }
    }

    // ========== Break/Continue/Return Tests ==========

    #[test]
    fn test_break_in_loop() {
        // break without value needs proper context (before ; or })
        let source = r#"
            fn test() {
                loop { break; }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_break_with_value() {
        let source = "break 42";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Break(value) => assert!(value.is_some()),
            _ => panic!("expected break expression"),
        }
    }

    #[test]
    fn test_continue_in_loop() {
        // continue needs proper context
        let source = r#"
            fn test() {
                while true { continue; }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_return_in_function() {
        // return without value needs proper context (before ; or })
        let source = r#"
            fn test() {
                return;
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_return_with_value() {
        let source = "return 42";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Return(value) => assert!(value.is_some()),
            _ => panic!("expected return expression"),
        }
    }

    #[test]
    fn test_return_with_expression() {
        let source = "return x + y";
        let expr = parse_expr(source);
        match expr.kind {
            ExprKind::Return(Some(value)) => {
                assert!(matches!(value.kind, ExprKind::Binary { op: BinaryOp::Add, .. }));
            }
            _ => panic!("expected return with binary expression"),
        }
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_function_with_if() {
        let source = r#"
            fn max(a: i64, b: i64) -> i64 {
                if a > b { a } else { b }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_function_with_while_loop() {
        let source = r#"
            fn sum_to_n(n: i64) -> i64 {
                let mut result = 0;
                let mut i = 0;
                while i <= n {
                    result = result + i;
                    i = i + 1;
                }
                return result;
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_function_with_for_loop() {
        let source = r#"
            fn process_items(items: [i64]) {
                for item in items {
                    handle(item);
                }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_function_with_match() {
        let source = r#"
            fn describe(x: i64) -> String {
                match x {
                    0 => "zero",
                    1 => "one",
                    _ => "many"
                }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_nested_control_flow() {
        let source = r#"
            fn complex(x: i64) -> i64 {
                let mut result = 0;
                for i in items {
                    if i > 0 {
                        while result < i {
                            result = result + 1;
                        }
                    } else {
                        break;
                    }
                }
                return result;
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_early_return() {
        let source = r#"
            fn find_first_positive(items: [i64]) -> i64 {
                for item in items {
                    if item > 0 {
                        return item;
                    }
                }
                return 0;
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }

    #[test]
    fn test_loop_with_conditional_break() {
        let source = r#"
            fn find_value(target: i64) -> i64 {
                let mut i = 0;
                loop {
                    if i == target {
                        break i;
                    }
                    i = i + 1;
                }
            }
        "#;
        let file = parse_source(source);
        assert_eq!(file.items.len(), 1);
    }
}
