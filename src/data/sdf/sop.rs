use std::ops::{Add, Mul, Neg};

use nalgebra::Vector3;

use crate::data::Dimension;

use super::prod::SDFExprProd;

#[derive(Clone, Default)]
pub(super) struct SDFExprSOP {
    pub(super) prods: Vec<SDFExprProd>,
}

impl SDFExprSOP {
    pub(super) fn eval(&self, at: &Vector3<f64>) -> f64 {
        self.prods.iter().map(|p| p.eval(at)).sum()
    }

    pub(super) fn derivative(&self, wrt: &Dimension) -> Self {
        let mut derivs = Self::default();
        for prod in &self.prods {
            derivs = derivs + prod.derivative(wrt);
        }

        derivs
    }
}

impl Add<Self> for SDFExprSOP {
    type Output = Self;

    fn add(mut self, mut rhs: Self) -> Self::Output {
        self.prods.append(&mut rhs.prods);
        self
    }
}

impl Neg for SDFExprSOP {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            prods: self.prods.into_iter().map(Neg::neg).collect(),
        }
    }
}

impl Mul for SDFExprSOP {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut expansion = Vec::new();
        for s_prod in &self.prods {
            for r_prod in &rhs.prods {
                expansion.push(s_prod.clone() * r_prod.clone());
            }
        }

        Self { prods: expansion }
    }
}

impl<T> From<T> for SDFExprSOP
where
    T: Into<f64>,
{
    fn from(value: T) -> Self {
        SDFExprProd::from(value).into()
    }
}

impl From<Dimension> for SDFExprSOP {
    fn from(value: Dimension) -> Self {
        SDFExprProd::from(value).into()
    }
}

impl From<SDFExprProd> for SDFExprSOP {
    fn from(value: SDFExprProd) -> Self {
        Self { prods: vec![value] }
    }
}
