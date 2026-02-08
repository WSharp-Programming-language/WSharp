//! Pretty printing for MIR.

use crate::mir::{
    AggregateKind, BasicBlockId, BinOp, BorrowKind, CastKind, Constant, Local, MirBody, MirModule,
    Mutability, NullOp, Operand, Place, PlaceElem, Rvalue, Statement, StatementKind,
    TerminatorKind, UnOp,
};
use std::fmt::{self, Write};

/// Pretty print a MIR module.
pub fn pretty_print_module(module: &MirModule) -> String {
    let mut output = String::new();
    let mut printer = MirPrinter::new(&mut output);
    printer.print_module(module);
    output
}

/// Pretty print a single MIR body.
pub fn pretty_print_body(body: &MirBody) -> String {
    let mut output = String::new();
    let mut printer = MirPrinter::new(&mut output);
    printer.print_body(body);
    output
}

struct MirPrinter<'a> {
    output: &'a mut String,
    indent: usize,
}

impl<'a> MirPrinter<'a> {
    fn new(output: &'a mut String) -> Self {
        Self { output, indent: 0 }
    }

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn writeln(&mut self, s: &str) {
        self.write_indent();
        self.output.push_str(s);
        self.output.push('\n');
    }

    fn write_indent(&mut self) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
    }

    fn print_module(&mut self, module: &MirModule) {
        self.writeln(&format!("// MIR Module: {}", module.name));
        if let Some(entry) = module.entry_point {
            self.writeln(&format!("// Entry point: body{}", entry.0));
        }
        self.write("\n");

        for (id, body) in &module.bodies {
            self.writeln(&format!("// Body ID: {}", id.0));
            self.print_body(body);
            self.write("\n");
        }
    }

    fn print_body(&mut self, body: &MirBody) {
        // Function signature
        let async_str = if body.is_async { "async " } else { "" };
        self.write(&format!("{}fn {}(", async_str, body.name));

        // Parameters (locals 1 to arg_count)
        for i in 0..body.arg_count {
            if i > 0 {
                self.write(", ");
            }
            let local = Local(i as u32 + 1);
            let decl = &body.locals[local.0 as usize];
            if let Some(ref name) = decl.name {
                self.write(&format!("{}: {}", name, decl.ty));
            } else {
                self.write(&format!("{}: {}", local, decl.ty));
            }
        }

        self.write(&format!(") -> {} {{\n", body.return_ty));
        self.indent += 1;

        // Local declarations
        for (i, decl) in body.locals.iter().enumerate() {
            let local = Local(i as u32);
            if i == 0 {
                self.writeln(&format!("let {}: {};  // return place", local, decl.ty));
            } else if i <= body.arg_count {
                // Skip arguments, already printed in signature
            } else {
                let name_comment = decl
                    .name
                    .as_ref()
                    .map(|n| format!("  // {}", n))
                    .unwrap_or_default();
                let mut_str = if decl.mutable { "mut " } else { "" };
                self.writeln(&format!(
                    "let {}{}: {};{}",
                    mut_str, local, decl.ty, name_comment
                ));
            }
        }

        if !body.locals.is_empty() {
            self.write("\n");
        }

        // Basic blocks
        for (id, block) in &body.basic_blocks {
            self.write_indent();
            self.write(&format!("{}: {{\n", id));
            self.indent += 1;

            for stmt in &block.statements {
                self.print_statement(stmt);
            }

            if let Some(ref term) = block.terminator {
                self.print_terminator(&term.kind);
            }

            self.indent -= 1;
            self.writeln("}");
            self.write("\n");
        }

        self.indent -= 1;
        self.writeln("}");
    }

    fn print_statement(&mut self, stmt: &Statement) {
        match &stmt.kind {
            StatementKind::Assign(place, rvalue) => {
                self.write_indent();
                self.print_place(place);
                self.write(" = ");
                self.print_rvalue(rvalue);
                self.write(";\n");
            }
            StatementKind::StorageLive(local) => {
                self.writeln(&format!("StorageLive({});", local));
            }
            StatementKind::StorageDead(local) => {
                self.writeln(&format!("StorageDead({});", local));
            }
            StatementKind::Nop => {
                self.writeln("nop;");
            }
        }
    }

    fn print_terminator(&mut self, kind: &TerminatorKind) {
        match kind {
            TerminatorKind::Goto { target } => {
                self.writeln(&format!("goto -> {};", target));
            }
            TerminatorKind::SwitchInt { discr, targets } => {
                self.write_indent();
                self.write("switchInt(");
                self.print_operand(discr);
                self.write(") -> [");

                for (i, (val, target)) in targets.targets.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.write(&format!("{}: {}", val, target));
                }

                self.write(&format!(", otherwise: {}];\n", targets.otherwise));
            }
            TerminatorKind::Return => {
                self.writeln("return;");
            }
            TerminatorKind::Unreachable => {
                self.writeln("unreachable;");
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
            } => {
                self.write_indent();
                self.print_place(destination);
                self.write(" = ");
                self.print_operand(func);
                self.write("(");

                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_operand(arg);
                }

                self.write(")");

                if let Some(t) = target {
                    self.write(&format!(" -> {};", t));
                } else {
                    self.write(" -> diverge;");
                }
                self.write("\n");
            }
            TerminatorKind::Drop { place, target } => {
                self.write_indent();
                self.write("drop(");
                self.print_place(place);
                self.write(&format!(") -> {};\n", target));
            }
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
            } => {
                self.write_indent();
                self.write(&format!("assert({} == {}, \"{}\") -> {};\n",
                    self.operand_to_string(cond), expected, msg, target));
            }
            TerminatorKind::Yield { value, resume } => {
                self.write_indent();
                self.write("yield(");
                self.print_operand(value);
                self.write(&format!(") -> {};\n", resume));
            }
        }
    }

    fn print_place(&mut self, place: &Place) {
        self.write(&format!("{}", place));
    }

    fn print_operand(&mut self, op: &Operand) {
        match op {
            Operand::Copy(place) => {
                self.write("copy ");
                self.print_place(place);
            }
            Operand::Move(place) => {
                self.write("move ");
                self.print_place(place);
            }
            Operand::Constant(c) => {
                self.print_constant(c);
            }
        }
    }

    fn operand_to_string(&self, op: &Operand) -> String {
        match op {
            Operand::Copy(place) => format!("copy {}", place),
            Operand::Move(place) => format!("move {}", place),
            Operand::Constant(c) => self.constant_to_string(c),
        }
    }

    fn print_constant(&mut self, c: &Constant) {
        self.write(&self.constant_to_string(c));
    }

    fn constant_to_string(&self, c: &Constant) -> String {
        match c {
            Constant::Int(v, ty) => format!("const {}_{}", v, ty),
            Constant::Float(v, ty) => format!("const {}_{}", v, ty),
            Constant::Bool(v) => format!("const {}", v),
            Constant::Char(c) => format!("const '{}'", c),
            Constant::String(s) => format!("const \"{}\"", s.escape_default()),
            Constant::Unit => "const ()".to_string(),
            Constant::Function(id) => format!("const fn#{}", id.0),
            Constant::Null => "const null".to_string(),
            Constant::Intrinsic(name) => format!("intrinsic @{name}"),
        }
    }

    fn print_rvalue(&mut self, rv: &Rvalue) {
        match rv {
            Rvalue::Use(op) => {
                self.print_operand(op);
            }
            Rvalue::Repeat(op, count) => {
                self.write("[");
                self.print_operand(op);
                self.write(&format!("; {}]", count));
            }
            Rvalue::Ref(place, kind) => {
                let kind_str = match kind {
                    BorrowKind::Shared => "&",
                    BorrowKind::Mut => "&mut ",
                };
                self.write(kind_str);
                self.print_place(place);
            }
            Rvalue::AddressOf(place, mutability) => {
                let mut_str = match mutability {
                    Mutability::Not => "const",
                    Mutability::Mut => "mut",
                };
                self.write(&format!("&raw {} ", mut_str));
                self.print_place(place);
            }
            Rvalue::Len(place) => {
                self.write("Len(");
                self.print_place(place);
                self.write(")");
            }
            Rvalue::BinaryOp(op, left, right) => {
                self.write(&format!("{:?}(", op));
                self.print_operand(left);
                self.write(", ");
                self.print_operand(right);
                self.write(")");
            }
            Rvalue::CheckedBinaryOp(op, left, right) => {
                self.write(&format!("Checked{:?}(", op));
                self.print_operand(left);
                self.write(", ");
                self.print_operand(right);
                self.write(")");
            }
            Rvalue::UnaryOp(op, operand) => {
                self.write(&format!("{:?}(", op));
                self.print_operand(operand);
                self.write(")");
            }
            Rvalue::NullaryOp(op, ty) => {
                self.write(&format!("{:?}({})", op, ty));
            }
            Rvalue::Cast(kind, op, ty) => {
                self.print_operand(op);
                self.write(&format!(" as {} ({:?})", ty, kind));
            }
            Rvalue::Discriminant(place) => {
                self.write("discriminant(");
                self.print_place(place);
                self.write(")");
            }
            Rvalue::Aggregate(kind, ops) => {
                match kind {
                    AggregateKind::Tuple => self.write("("),
                    AggregateKind::Array(ty) => self.write(&format!("[{};", ty)),
                    AggregateKind::Adt { name, variant } => {
                        self.write(name);
                        if let Some(v) = variant {
                            self.write(&format!("::variant#{}", v));
                        }
                        self.write(" { ");
                    }
                    AggregateKind::Closure { body_id, .. } => {
                        self.write(&format!("[closure@{}](", body_id.0));
                    }
                }

                for (i, op) in ops.iter().enumerate() {
                    if i > 0 {
                        self.write(", ");
                    }
                    self.print_operand(op);
                }

                match kind {
                    AggregateKind::Tuple => self.write(")"),
                    AggregateKind::Array(_) => self.write("]"),
                    AggregateKind::Adt { .. } => self.write(" }"),
                    AggregateKind::Closure { .. } => self.write(")"),
                }
            }
            Rvalue::ShallowInitBox(op, ty) => {
                self.write(&format!("ShallowInitBox::<{}>(", ty));
                self.print_operand(op);
                self.write(")");
            }
        }
    }
}
