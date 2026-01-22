//! Statement parsing.

use crate::error::ParseResult;
use crate::parser::Parser;
use wsharp_ast::*;
use wsharp_lexer::TokenKind;

impl Parser<'_> {
    pub fn parse_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.current_span();

        // Let statement
        if self.check(&TokenKind::Let) {
            return self.parse_let_stmt();
        }

        // Empty statement
        if self.match_token(&TokenKind::Semi) {
            let id = self.next_id();
            return Ok(Stmt::new(StmtKind::Empty, start, id));
        }

        // Expression statement
        let expr = self.parse_expr()?;

        // Check if this needs a semicolon
        let needs_semi = !matches!(
            expr.kind,
            ExprKind::If { .. }
                | ExprKind::Match { .. }
                | ExprKind::While { .. }
                | ExprKind::For { .. }
                | ExprKind::Loop { .. }
                | ExprKind::Block(_)
        );

        if needs_semi && !self.check(&TokenKind::RBrace) {
            self.expect(TokenKind::Semi)?;
        }

        let id = self.next_id();
        Ok(Stmt::new(StmtKind::Expr(expr), start.merge(self.current_span()), id))
    }

    fn parse_let_stmt(&mut self) -> ParseResult<Stmt> {
        let start = self.current_span();
        self.expect(TokenKind::Let)?;

        let mutable = self.match_token(&TokenKind::Mut);
        let name = self.expect_ident()?;

        let ty = if self.match_token(&TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let value = if self.match_token(&TokenKind::Eq) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        self.expect(TokenKind::Semi)?;

        let id = self.next_id();
        Ok(Stmt::new(
            StmtKind::Let {
                name,
                ty,
                value,
                mutable,
            },
            start.merge(self.current_span()),
            id,
        ))
    }
}
