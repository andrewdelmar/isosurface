use std::{
    fmt::Display,
    marker::ConstParamTy,
    ops::{self, Neg},
};

use nalgebra::Vector3;

#[derive(Clone, Copy, PartialEq, Eq, ConstParamTy)]
pub enum Dimension {
    X,
    Y,
    Z,
}

impl Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dimension::X => write!(f, "X"),
            Dimension::Y => write!(f, "Y"),
            Dimension::Z => write!(f, "Z"),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Condition {
    Greater,
    Less,
}

impl Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Condition::Greater => write!(f, ">"),
            Condition::Less => write!(f, "<"),
        }
    }
}

#[derive(Clone)]
pub enum SDFExpression {
    Const(f64),
    Dim(Dimension),
    Add(Box<SDFExpression>, Box<SDFExpression>), //TODO this and Mul should just be a SOP collection.
    Mul(Box<SDFExpression>, Box<SDFExpression>),
    Max(Box<SDFExpression>, Box<SDFExpression>),
    Min(Box<SDFExpression>, Box<SDFExpression>),
    When {
        cond: Condition,
        condl: Box<SDFExpression>,
        condr: Box<SDFExpression>,
        vall: Box<SDFExpression>,
        valr: Box<SDFExpression>,
    },
}

impl SDFExpression {
    pub const X: SDFExpression = SDFExpression::Dim(Dimension::X);
    pub const Y: SDFExpression = SDFExpression::Dim(Dimension::Y);
    pub const Z: SDFExpression = SDFExpression::Dim(Dimension::Z);

    pub(crate) fn eval(&self, at: &Vector3<f64>) -> f64 {
        match &self {
            SDFExpression::Const(t) => *t,
            SDFExpression::Dim(d) => match d {
                Dimension::X => at.x,
                Dimension::Y => at.y,
                Dimension::Z => at.z,
            },
            SDFExpression::Add(u, v) => u.eval(at) + v.eval(at),
            SDFExpression::Mul(u, v) => u.eval(at) * v.eval(at),
            SDFExpression::Min(u, v) => f64::min(u.eval(at), v.eval(at)),
            SDFExpression::Max(u, v) => f64::max(u.eval(at), v.eval(at)),
            SDFExpression::When {
                cond,
                condl,
                condr,
                vall,
                valr,
            } => {
                let l = condl.eval(at);
                let r = condr.eval(at);

                if (l > r && cond == &Condition::Greater) || (l < r && cond == &Condition::Less) {
                    vall.eval(at)
                } else {
                    valr.eval(at)
                }
            }
        }
    }

    pub(crate) fn derive(&self, to: &Dimension) -> Self {
        match self {
            SDFExpression::Const(_) => Self::Const(0.0),
            SDFExpression::Dim(d) => {
                if d == to {
                    Self::Const(1.0)
                } else {
                    Self::Const(0.0)
                }
            }
            SDFExpression::Add(box u, box v) => u.derive(to) + v.derive(to),
            SDFExpression::Mul(box u, box v) => u.derive(to) * v.clone() + v.derive(to) * u.clone(),
            SDFExpression::Max(u, v) => Self::When {
                cond: Condition::Greater,
                condl: u.clone(),
                condr: v.clone(),
                vall: Box::new(u.derive(to)),
                valr: Box::new(v.derive(to)),
            },
            SDFExpression::Min(u, v) => Self::When {
                cond: Condition::Less,
                condl: u.clone(),
                condr: v.clone(),
                vall: Box::new(u.derive(to)),
                valr: Box::new(v.derive(to)),
            },
            SDFExpression::When {
                cond,
                condl,
                condr,
                vall,
                valr,
            } => Self::When {
                cond: cond.clone(),
                condl: condl.clone(),
                condr: condr.clone(),
                vall: Box::new(vall.derive(to)),
                valr: Box::new(valr.derive(to)),
            },
        }
    }
}

impl ops::Neg for SDFExpression {
    type Output = SDFExpression;

    fn neg(self) -> Self::Output {
        SDFExpression::Const(-1.0) * self
    }
}

impl ops::Add for SDFExpression {
    type Output = SDFExpression;

    fn add(self, rhs: Self) -> Self::Output {
        SDFExpression::Add(Box::new(self), Box::new(rhs.clone()))
    }
}

impl ops::Sub for SDFExpression {
    type Output = SDFExpression;

    fn sub(self, rhs: Self) -> Self::Output {
        SDFExpression::Add(Box::new(self), Box::new(rhs.neg()))
    }
}

impl ops::Mul for SDFExpression {
    type Output = SDFExpression;

    fn mul(self, rhs: Self) -> Self::Output {
        SDFExpression::Mul(Box::new(self), Box::new(rhs.clone()))
    }
}

impl From<f64> for SDFExpression {
    fn from(value: f64) -> Self {
        Self::Const(value)
    }
}

impl From<Dimension> for SDFExpression {
    fn from(value: Dimension) -> Self {
        SDFExpression::Dim(value)
    }
}

impl Display for SDFExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SDFExpression::Const(c) => write!(f, "{}", c),
            SDFExpression::Dim(d) => write!(f, "{}", d),
            SDFExpression::Add(a, b) => write!(f, "{} + {}", a, b),
            SDFExpression::Mul(box a, box b) => {
                if let Self::Add(_, _) = a {
                    write!(f, "({})", a)?
                } else {
                    write!(f, "{}", a)?
                };
                write!(f, " * ")?;
                if let Self::Add(_, _) = b {
                    write!(f, "({})", b)
                } else {
                    write!(f, "{}", b)
                }
            }
            SDFExpression::Max(a, b) => write!(f, "max({}, {})", a, b),
            SDFExpression::Min(a, b) => write!(f, "min({}, {})", a, b),
            SDFExpression::When {
                cond,
                condl,
                condr,
                vall,
                valr,
            } => write!(
                f,
                "when({} {} {}, {} else {})",
                condl, cond, condr, vall, valr
            ),
        }
    }
}
