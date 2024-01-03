use std::ops::{Mul, Neg};

use nalgebra::Vector3;

use crate::data::Dimension;

use super::{sop::SDFExprSOP, term::SDFExprTerm};

#[derive(Clone)]
pub(super) struct SDFExprProd {
    pub(super) mul: f64,
    pub(super) terms: Vec<SDFExprTerm>,
}

impl SDFExprProd {
    pub(super) fn eval(&self, at: &Vector3<f64>) -> f64 {
        self.mul * self.terms.iter().map(|t| t.eval(at)).product::<f64>()
    }

    pub(super) fn derivative(&self, wrt: &Dimension) -> SDFExprSOP {
        let mut derivs = Vec::new();
        for (i, term) in self.terms.iter().enumerate() {
            if let Some(mut dt) = term.derivative(wrt) {
                let mut others = self.terms.clone();
                others.remove(i);
                dt.terms.append(&mut others);
                derivs.push(dt);
            }
        }

        SDFExprSOP { prods: derivs }
    }
}

impl Neg for SDFExprProd {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.mul *= -1.0;
        self
    }
}

impl Mul for SDFExprProd {
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        self.mul *= rhs.mul;
        self.terms.append(&mut rhs.terms);
        self
    }
}

impl<T> From<T> for SDFExprProd
where
    T: Into<f64>,
{
    fn from(value: T) -> Self {
        Self {
            mul: value.into(),
            terms: Vec::default(),
        }
    }
}

impl From<Dimension> for SDFExprProd {
    fn from(value: Dimension) -> Self {
        SDFExprTerm::from(value).into()
    }
}

impl From<SDFExprTerm> for SDFExprProd {
    fn from(value: SDFExprTerm) -> Self {
        Self {
            mul: 1.0,
            terms: vec![value],
        }
    }
}
