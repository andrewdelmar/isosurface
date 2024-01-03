use std::{
    cell::RefCell,
    ops::{Add, Mul, Neg, Sub},
    rc::Rc,
};

use nalgebra::Vector3;

use crate::VolumetricFunc;

use super::Dimension;

#[derive(Clone)]
pub struct SDFExpression {
    sops: SDFExprSOP,
    deriv_cache: RefCell<Option<Box<[SDFExprSOP; 3]>>>,
}

impl From<f64> for SDFExpression {
    fn from(value: f64) -> Self {
        Self::constant(value)
    }
}

impl Add<SDFExpression> for SDFExpression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut sops = self.sops;
        let mut rhs = rhs.sops;
        sops.append(&mut rhs);
        Self {
            sops,
            deriv_cache: RefCell::new(None),
        }
    }
}

impl Neg for SDFExpression {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for prod in &mut self.sops {
            prod.mul = -prod.mul;
        }
        self
    }
}

impl Sub<SDFExpression> for SDFExpression {
    type Output = Self;

    fn sub(self, rhs: SDFExpression) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul<SDFExpression> for SDFExpression {
    type Output = Self;

    fn mul(self, rhs: SDFExpression) -> Self::Output {
        todo!()
    }
}

impl SDFExpression {
    const X: Self = SDFExprTerm::Dim(Dimension::X).into();
    const Y: Self = SDFExprTerm::Dim(Dimension::Y).into();
    const Z: Self = SDFExprTerm::Dim(Dimension::Z).into();

    fn max(a: Self, b: Self) -> Self {
        let (a, b) = (Rc::new(a), Rc::new(b));
        SDFExprTerm::GT {
            left: a.clone(),
            right: b.clone(),
            true_val: a,
            false_val: b,
        }
        .into()
    }

    fn min(a: Self, b: Self) -> Self {
        let (a, b) = (Rc::new(a), Rc::new(b));
        SDFExprTerm::GT {
            left: a.clone(),
            right: b.clone(),
            true_val: b,
            false_val: a,
        }
        .into()
    }

    fn constant(value: f64) -> Self {
        Self {
            sops: vec![SDFExprProd {
                mul: value,
                terms: vec![SDFExprTerm::Unit],
            }],
            deriv_cache: RefCell::new(None),
        }
    }

    fn derivative(&self, wrt: &Dimension) -> Vec<SDFExprProd> {
        self.sops.iter().flat_map(|p| p.derivative(wrt)).collect()
    }
}


/*
impl From<SDFExprTerm> for SDFExpression {
    fn from(value: SDFExprTerm) -> Self {
        Self {
            sops: value.into(),
            deriv_cache: RefCell::new(None),
        }
    }
} */

impl VolumetricFunc for SDFExpression {
    fn eval(&self, at: &Vector3<f64>) -> f64 {
        self.sops.iter().map(|p| p.eval(at)).sum()
    }

    fn grad(&self, at: &Vector3<f64>) -> Vector3<f64> {
        let del = self.deriv_cache.borrow_mut().get_or_insert_with(|| {
            Box::new([
                self.derivative(&Dimension::X),
                self.derivative(&Dimension::Y),
                self.derivative(&Dimension::Z),
            ])
        });

        let vals = del.map(|d| d.iter().map(|p| p.eval(at)).sum());
        vals.into()
    }
}

#[derive(Clone)]
struct SDFExprSOP {
    prods: Vec<SDFExprProd>,
}

#[derive(Clone)]
struct SDFExprProd {
    mul: f64,
    terms: Vec<SDFExprTerm>,
}

impl From<SDFExprTerm> for SDFExprProd {
    fn from(value: SDFExprTerm) -> Self {
        Self {
            mul: 1.0,
            terms: vec![value],
        }
    }
}


