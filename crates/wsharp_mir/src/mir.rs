//! Mid-level Intermediate Representation definitions.
//!
//! MIR is a control-flow graph based representation in SSA form.
//! It's designed for optimization and code generation.

use indexmap::IndexMap;
use std::fmt;
use wsharp_types::Type;

/// A unique identifier for a MIR body (function).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyId(pub u32);

/// A unique identifier for a basic block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BasicBlockId(pub u32);

impl BasicBlockId {
    pub const ENTRY: BasicBlockId = BasicBlockId(0);
}

impl fmt::Display for BasicBlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// A unique identifier for a local variable in MIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Local(pub u32);

impl Local {
    /// The return place is always local 0.
    pub const RETURN_PLACE: Local = Local(0);
}

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "_0")
        } else {
            write!(f, "_{}", self.0)
        }
    }
}

/// A complete MIR module containing multiple function bodies.
#[derive(Clone, Debug)]
pub struct MirModule {
    pub name: String,
    pub bodies: IndexMap<BodyId, MirBody>,
    pub entry_point: Option<BodyId>,
}

impl MirModule {
    pub fn new(name: String) -> Self {
        Self {
            name,
            bodies: IndexMap::new(),
            entry_point: None,
        }
    }

    pub fn add_body(&mut self, body: MirBody) -> BodyId {
        let id = BodyId(self.bodies.len() as u32);
        self.bodies.insert(id, body);
        id
    }
}

/// A MIR function body - the control flow graph for a single function.
#[derive(Clone, Debug)]
pub struct MirBody {
    /// The function name.
    pub name: String,

    /// Local variable declarations (index 0 is the return place).
    pub locals: Vec<LocalDecl>,

    /// The number of arguments (first N locals after return place).
    pub arg_count: usize,

    /// Basic blocks forming the control flow graph.
    pub basic_blocks: IndexMap<BasicBlockId, BasicBlock>,

    /// The return type.
    pub return_ty: Type,

    /// Whether this is an async function.
    pub is_async: bool,
}

impl MirBody {
    pub fn new(name: String, return_ty: Type, is_async: bool) -> Self {
        // Local 0 is always the return place
        let return_place = LocalDecl {
            ty: return_ty.clone(),
            name: None,
            mutable: true,
        };

        Self {
            name,
            locals: vec![return_place],
            arg_count: 0,
            basic_blocks: IndexMap::new(),
            return_ty,
            is_async,
        }
    }

    /// Add a new local variable and return its index.
    pub fn add_local(&mut self, ty: Type, name: Option<String>, mutable: bool) -> Local {
        let local = Local(self.locals.len() as u32);
        self.locals.push(LocalDecl { ty, name, mutable });
        local
    }

    /// Add an argument local.
    pub fn add_arg(&mut self, ty: Type, name: String) -> Local {
        let local = self.add_local(ty, Some(name), false);
        self.arg_count += 1;
        local
    }

    /// Create a new basic block and return its ID.
    pub fn new_basic_block(&mut self) -> BasicBlockId {
        let id = BasicBlockId(self.basic_blocks.len() as u32);
        self.basic_blocks.insert(id, BasicBlock::new());
        id
    }

    /// Get a mutable reference to a basic block.
    pub fn block_mut(&mut self, id: BasicBlockId) -> &mut BasicBlock {
        self.basic_blocks.get_mut(&id).expect("invalid block id")
    }

    /// Get a reference to a basic block.
    pub fn block(&self, id: BasicBlockId) -> &BasicBlock {
        self.basic_blocks.get(&id).expect("invalid block id")
    }

    /// Get the entry block (always bb0).
    pub fn entry_block(&self) -> BasicBlockId {
        BasicBlockId::ENTRY
    }
}

/// Declaration of a local variable.
#[derive(Clone, Debug)]
pub struct LocalDecl {
    pub ty: Type,
    pub name: Option<String>,
    pub mutable: bool,
}

/// A basic block in the control flow graph.
#[derive(Clone, Debug)]
pub struct BasicBlock {
    /// Statements executed in order.
    pub statements: Vec<Statement>,

    /// The terminator that ends this block.
    pub terminator: Option<Terminator>,
}

impl BasicBlock {
    pub fn new() -> Self {
        Self {
            statements: Vec::new(),
            terminator: None,
        }
    }

    pub fn push_stmt(&mut self, stmt: Statement) {
        self.statements.push(stmt);
    }

    pub fn set_terminator(&mut self, term: Terminator) {
        self.terminator = Some(term);
    }
}

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// A statement (non-terminating instruction).
#[derive(Clone, Debug)]
pub struct Statement {
    pub kind: StatementKind,
}

/// The kind of statement.
#[derive(Clone, Debug)]
pub enum StatementKind {
    /// Assign an rvalue to a place.
    Assign(Place, Rvalue),

    /// Storage live marker (for stack slot allocation).
    StorageLive(Local),

    /// Storage dead marker (for stack slot deallocation).
    StorageDead(Local),

    /// No-op (used for debugging or removed optimizations).
    Nop,
}

/// A terminator instruction that ends a basic block.
#[derive(Clone, Debug)]
pub struct Terminator {
    pub kind: TerminatorKind,
}

/// The kind of terminator.
#[derive(Clone, Debug)]
pub enum TerminatorKind {
    /// Unconditional jump to a block.
    Goto {
        target: BasicBlockId,
    },

    /// Conditional branch.
    SwitchInt {
        discr: Operand,
        targets: SwitchTargets,
    },

    /// Return from the function.
    Return,

    /// Unreachable code (e.g., after infinite loop or panic).
    Unreachable,

    /// Function call.
    Call {
        func: Operand,
        args: Vec<Operand>,
        destination: Place,
        target: Option<BasicBlockId>, // None for diverging calls
    },

    /// Drop a value (run destructor).
    Drop {
        place: Place,
        target: BasicBlockId,
    },

    /// Assert a condition, panic if false.
    Assert {
        cond: Operand,
        expected: bool,
        msg: String,
        target: BasicBlockId,
    },

    /// Yield for async/generator functions.
    Yield {
        value: Operand,
        resume: BasicBlockId,
    },
}

/// Switch targets for SwitchInt terminator.
#[derive(Clone, Debug)]
pub struct SwitchTargets {
    /// (value, target) pairs for specific values.
    pub targets: Vec<(u128, BasicBlockId)>,
    /// Default target if no value matches.
    pub otherwise: BasicBlockId,
}

impl SwitchTargets {
    pub fn new(targets: Vec<(u128, BasicBlockId)>, otherwise: BasicBlockId) -> Self {
        Self { targets, otherwise }
    }

    /// Create a simple if-else switch (0 = else, otherwise = then).
    pub fn if_else(then_block: BasicBlockId, else_block: BasicBlockId) -> Self {
        Self {
            targets: vec![(0, else_block)], // false (0) goes to else
            otherwise: then_block,          // true (non-zero) goes to then
        }
    }
}

/// A place in memory (lvalue).
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub local: Local,
    pub projection: Vec<PlaceElem>,
}

impl Place {
    pub fn local(local: Local) -> Self {
        Self {
            local,
            projection: Vec::new(),
        }
    }

    pub fn return_place() -> Self {
        Self::local(Local::RETURN_PLACE)
    }

    pub fn project(mut self, elem: PlaceElem) -> Self {
        self.projection.push(elem);
        self
    }

    pub fn field(self, field: usize, ty: Type) -> Self {
        self.project(PlaceElem::Field(field, ty))
    }

    pub fn index(self, index: Local) -> Self {
        self.project(PlaceElem::Index(index))
    }

    pub fn deref(self) -> Self {
        self.project(PlaceElem::Deref)
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.local)?;
        for elem in &self.projection {
            match elem {
                PlaceElem::Deref => write!(f, ".*")?,
                PlaceElem::Field(idx, _) => write!(f, ".{}", idx)?,
                PlaceElem::Index(local) => write!(f, "[{}]", local)?,
                PlaceElem::ConstantIndex { offset, from_end, .. } => {
                    if *from_end {
                        write!(f, "[-{}]", offset)?;
                    } else {
                        write!(f, "[{}]", offset)?;
                    }
                }
                PlaceElem::Subslice { from, to, from_end } => {
                    if *from_end {
                        write!(f, "[{}..-{}]", from, to)?;
                    } else {
                        write!(f, "[{}..{}]", from, to)?;
                    }
                }
                PlaceElem::Downcast(idx) => write!(f, " as variant#{}", idx)?,
            }
        }
        Ok(())
    }
}

/// A projection element for places.
#[derive(Clone, Debug, PartialEq)]
pub enum PlaceElem {
    /// Dereference a pointer/reference.
    Deref,

    /// Access a field by index.
    Field(usize, Type),

    /// Index by a local variable.
    Index(Local),

    /// Index by a constant.
    ConstantIndex {
        offset: usize,
        min_length: usize,
        from_end: bool,
    },

    /// Slice operation.
    Subslice {
        from: usize,
        to: usize,
        from_end: bool,
    },

    /// Downcast to a variant (for enums).
    Downcast(usize),
}

/// An operand (rvalue that can be used directly).
#[derive(Clone, Debug)]
pub enum Operand {
    /// Copy the value from a place.
    Copy(Place),

    /// Move the value from a place.
    Move(Place),

    /// A constant value.
    Constant(Constant),
}

impl Operand {
    pub fn const_bool(value: bool) -> Self {
        Operand::Constant(Constant::Bool(value))
    }

    pub fn const_int(value: i128, ty: Type) -> Self {
        Operand::Constant(Constant::Int(value, ty))
    }

    pub fn const_float(value: f64, ty: Type) -> Self {
        Operand::Constant(Constant::Float(value, ty))
    }

    pub fn const_unit() -> Self {
        Operand::Constant(Constant::Unit)
    }
}

/// A constant value.
#[derive(Clone, Debug)]
pub enum Constant {
    /// Integer constant with type.
    Int(i128, Type),

    /// Float constant with type.
    Float(f64, Type),

    /// Boolean constant.
    Bool(bool),

    /// Character constant.
    Char(char),

    /// String constant.
    String(String),

    /// Unit value.
    Unit,

    /// Function reference.
    Function(BodyId),

    /// Null pointer.
    Null,
}

/// An rvalue (right-hand side of an assignment).
#[derive(Clone, Debug)]
pub enum Rvalue {
    /// Use an operand directly.
    Use(Operand),

    /// Repeat a value N times (array initialization).
    Repeat(Operand, usize),

    /// Create a reference to a place.
    Ref(Place, BorrowKind),

    /// Get the address of a place (raw pointer).
    AddressOf(Place, Mutability),

    /// Get the length of a slice/array.
    Len(Place),

    /// Binary operation.
    BinaryOp(BinOp, Operand, Operand),

    /// Checked binary operation (returns (result, overflow) tuple).
    CheckedBinaryOp(BinOp, Operand, Operand),

    /// Unary operation.
    UnaryOp(UnOp, Operand),

    /// Null-unary operation (just negation of null check).
    NullaryOp(NullOp, Type),

    /// Type cast.
    Cast(CastKind, Operand, Type),

    /// Discriminant of an enum.
    Discriminant(Place),

    /// Create an aggregate value (tuple, array, struct).
    Aggregate(AggregateKind, Vec<Operand>),

    /// Shallow initialization check.
    ShallowInitBox(Operand, Type),
}

/// Kind of borrow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorrowKind {
    /// Shared (immutable) borrow.
    Shared,
    /// Mutable borrow.
    Mut,
}

/// Mutability marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutability {
    Not,
    Mut,
}

/// Binary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Pointer offset
    Offset,
}

/// Unary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOp {
    /// Logical/bitwise not.
    Not,
    /// Arithmetic negation.
    Neg,
}

/// Nullary operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullOp {
    /// Size of a type.
    SizeOf,
    /// Alignment of a type.
    AlignOf,
}

/// Type cast kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastKind {
    /// Numeric cast (int to float, etc).
    IntToInt,
    IntToFloat,
    FloatToInt,
    FloatToFloat,

    /// Pointer casts.
    PtrToPtr,
    FnPtrToPtr,

    /// Pointer/int casts (unsafe).
    PointerExposeAddress,
    PointerFromExposedAddress,
}

/// Aggregate value kinds.
#[derive(Clone, Debug)]
pub enum AggregateKind {
    /// Tuple.
    Tuple,

    /// Array.
    Array(Type),

    /// Struct/prototype.
    Adt {
        name: String,
        variant: Option<usize>,
    },

    /// Closure.
    Closure {
        body_id: BodyId,
        captures: Vec<Type>,
    },
}
