mod prod;
use prod::SDFExprProd;

mod sop;
use sop::SDFExprSOP;

mod term;
use term::SDFExprTerm;

use crate::{Dimension, VolumetricFunc};
use nalgebra::Vector3;
use std::{
    ops::{Add, Mul, Neg, Sub},
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
pub struct SDFExpression {
    sops: SDFExprSOP,
    grad_cache: Arc<Mutex<Option<[SDFExprSOP; 3]>>>,
}

impl VolumetricFunc for SDFExpression {
    fn eval(&self, at: &nalgebra::Vector3<f64>) -> f64 {
        self.sops.eval(at)
    }

    fn grad(&self, at: &nalgebra::Vector3<f64>) -> Vector3<f64> {
        let mut cache = self.grad_cache.lock().unwrap();
        let [x, y, z] = cache.get_or_insert_with(|| self.derive_grad());

        Vector3::new(x.eval(at), y.eval(at), z.eval(at))
    }
}

impl SDFExpression {
    fn derive_grad(&self) -> [SDFExprSOP; 3] {
        [
            self.sops.derivative(&super::Dimension::X),
            self.sops.derivative(&super::Dimension::Y),
            self.sops.derivative(&super::Dimension::Z),
        ]
    }

    pub fn max(a: Self, b: Self) -> Self {
        let arc = Arc::new(a.sops);
        let brc = Arc::new(b.sops);
        let term = SDFExprTerm::GT {
            left: arc.clone(),
            right: brc.clone(),
            true_val: arc,
            false_val: brc,
        };
        SDFExprSOP::from(SDFExprProd::from(term)).into()
    }

    pub fn min(a: Self, b: Self) -> Self {
        let arc = Arc::new(a.sops);
        let brc = Arc::new(b.sops);
        let term = SDFExprTerm::GT {
            left: arc.clone(),
            right: brc.clone(),
            true_val: brc,
            false_val: arc,
        };
        SDFExprSOP::from(SDFExprProd::from(term)).into()
    }

    pub fn x() -> Self {
        Dimension::X.into()
    }

    pub fn y() -> Self {
        Dimension::Y.into()
    }

    pub fn z() -> Self {
        Dimension::Z.into()
    }
}

impl Add<Self> for SDFExpression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        (self.sops + rhs.sops).into()
    }
}

impl Neg for SDFExpression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        (-self.sops).into()
    }
}

impl Sub for SDFExpression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul<Self> for SDFExpression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        (self.sops * rhs.sops).into()
    }
}

impl<T> From<T> for SDFExpression
where
    T: Into<f64>,
{
    fn from(value: T) -> Self {
        SDFExprSOP::from(value).into()
    }
}

impl From<Dimension> for SDFExpression {
    fn from(value: Dimension) -> Self {
        SDFExprSOP::from(value).into()
    }
}

impl From<SDFExprSOP> for SDFExpression {
    fn from(value: SDFExprSOP) -> Self {
        Self {
            sops: value,
            grad_cache: Arc::new(Mutex::default()),
        }
    }
}
