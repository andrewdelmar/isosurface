pub(crate) mod sdf;

use std::{fmt::Display, marker::ConstParamTy};
use nalgebra::Vector3;

pub trait VolumetricFunc {
    fn eval(&self, at: &Vector3<f64>) -> f64;
    fn grad(&self, at: &Vector3<f64>) -> Vector3<f64>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ConstParamTy, Hash)]
pub enum Dimension {
    X,
    Y,
    Z,
}

impl Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Dimension::X => "X",
            Dimension::Y => "Y",
            Dimension::Z => "Z",
        };
        write!(f, "{}", s)
    }
}
