//! Core type definitions for W#.

use std::fmt;

/// A unique identifier for a type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub u32);

/// The core type representation for W#.
#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    /// Primitive types
    Primitive(PrimitiveType),

    /// Array type with optional size
    Array {
        element: Box<Type>,
        size: Option<usize>,
    },

    /// Slice type (dynamically-sized view into array)
    Slice { element: Box<Type> },

    /// Tuple type
    Tuple(Vec<Type>),

    /// Function type
    Function(FunctionType),

    /// Prototype type (for prototype-based OOP)
    Prototype(PrototypeType),

    /// HTTP status type
    HttpStatus(HttpStatusType),

    /// Type variable (for inference)
    TypeVar(TypeVarId),

    /// Applied generic type
    Applied {
        base: Box<Type>,
        args: Vec<Type>,
    },

    /// Reference type
    Ref {
        inner: Box<Type>,
        mutable: bool,
    },

    /// Reference-counted type
    Rc(Box<Type>),

    /// Future type (for async)
    Future(Box<Type>),

    /// Never type (for functions that don't return)
    Never,

    /// Unknown type (for inference)
    Unknown,

    /// Unit type
    Unit,
}

/// Primitive types in W#.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveType {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Bool,
    Char,
    Str,
}

impl PrimitiveType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PrimitiveType::I8 => "i8",
            PrimitiveType::I16 => "i16",
            PrimitiveType::I32 => "i32",
            PrimitiveType::I64 => "i64",
            PrimitiveType::I128 => "i128",
            PrimitiveType::U8 => "u8",
            PrimitiveType::U16 => "u16",
            PrimitiveType::U32 => "u32",
            PrimitiveType::U64 => "u64",
            PrimitiveType::U128 => "u128",
            PrimitiveType::F32 => "f32",
            PrimitiveType::F64 => "f64",
            PrimitiveType::Bool => "bool",
            PrimitiveType::Char => "char",
            PrimitiveType::Str => "str",
        }
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integer() || self.is_float()
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            PrimitiveType::I8
                | PrimitiveType::I16
                | PrimitiveType::I32
                | PrimitiveType::I64
                | PrimitiveType::I128
                | PrimitiveType::U8
                | PrimitiveType::U16
                | PrimitiveType::U32
                | PrimitiveType::U64
                | PrimitiveType::U128
        )
    }

    pub fn is_float(&self) -> bool {
        matches!(self, PrimitiveType::F32 | PrimitiveType::F64)
    }
}

/// A type variable identifier for type inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeVarId(pub u32);

impl fmt::Display for TypeVarId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

/// Function type with dispatch information.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionType {
    pub params: Vec<ParamType>,
    pub return_type: Box<Type>,
    pub is_async: bool,
}

/// A parameter type with dispatch role information.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamType {
    pub ty: Type,
    pub dispatch_role: DispatchRole,
}

/// The role of a parameter in dispatch resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchRole {
    /// Not used for dispatch
    Static,
    /// Runtime dispatch based on actual type
    Dynamic,
    /// Compile-time specialization
    Specialized,
    /// Function dispatch (based on function identity)
    FunctionDispatch,
    /// HTTP status dispatch
    HttpStatusDispatch,
}

/// Prototype type for prototype-based OOP.
#[derive(Clone, Debug, PartialEq)]
pub struct PrototypeType {
    pub name: Option<String>,
    pub parent: Option<Box<Type>>,
    pub members: Vec<(String, Type)>,
}

/// HTTP status type.
#[derive(Clone, Debug, PartialEq)]
pub struct HttpStatusType {
    pub kind: HttpStatusTypeKind,
}

/// The kind of HTTP status type.
#[derive(Clone, Debug, PartialEq)]
pub enum HttpStatusTypeKind {
    /// Exact status code (e.g., 200, 404)
    Exact(u16),
    /// Category (e.g., 2xx success codes)
    Category(StatusCategory),
    /// Range (e.g., 400..499)
    Range { start: u16, end: u16 },
    /// Union of statuses
    Union(Vec<HttpStatusTypeKind>),
    /// Any HTTP status
    Any,
}

/// HTTP status code categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatusCategory {
    /// 100-199
    Informational,
    /// 200-299
    Success,
    /// 300-399
    Redirection,
    /// 400-499
    ClientError,
    /// 500-599
    ServerError,
}

impl StatusCategory {
    pub fn contains(&self, code: u16) -> bool {
        match self {
            StatusCategory::Informational => (100..200).contains(&code),
            StatusCategory::Success => (200..300).contains(&code),
            StatusCategory::Redirection => (300..400).contains(&code),
            StatusCategory::ClientError => (400..500).contains(&code),
            StatusCategory::ServerError => (500..600).contains(&code),
        }
    }

    pub fn from_code(code: u16) -> Option<StatusCategory> {
        match code {
            100..200 => Some(StatusCategory::Informational),
            200..300 => Some(StatusCategory::Success),
            300..400 => Some(StatusCategory::Redirection),
            400..500 => Some(StatusCategory::ClientError),
            500..600 => Some(StatusCategory::ServerError),
            _ => None,
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Primitive(p) => write!(f, "{}", p.as_str()),
            Type::Array { element, size } => {
                if let Some(s) = size {
                    write!(f, "[{}; {}]", element, s)
                } else {
                    write!(f, "[{}]", element)
                }
            }
            Type::Slice { element } => write!(f, "[{}]", element),
            Type::Tuple(types) => {
                write!(f, "(")?;
                for (i, t) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            Type::Function(func) => {
                write!(f, "fn(")?;
                for (i, p) in func.params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.ty)?;
                }
                write!(f, ") -> {}", func.return_type)
            }
            Type::Prototype(proto) => {
                if let Some(name) = &proto.name {
                    write!(f, "{}", name)
                } else {
                    write!(f, "proto")
                }
            }
            Type::HttpStatus(status) => write!(f, "HttpStatus<{:?}>", status.kind),
            Type::TypeVar(id) => write!(f, "T{}", id.0),
            Type::Applied { base, args } => {
                write!(f, "{}<", base)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ">")
            }
            Type::Ref { inner, mutable } => {
                if *mutable {
                    write!(f, "&mut {}", inner)
                } else {
                    write!(f, "&{}", inner)
                }
            }
            Type::Rc(inner) => write!(f, "Rc<{}>", inner),
            Type::Future(inner) => write!(f, "Future<{}>", inner),
            Type::Never => write!(f, "!"),
            Type::Unknown => write!(f, "?"),
            Type::Unit => write!(f, "()"),
        }
    }
}
