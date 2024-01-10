pub(crate) mod sdf;

use nalgebra::{SVector, Vector3};
use std::{fmt::Display, marker::ConstParamTy};

use crate::subspace::Subspace;

pub trait VolumetricFunc: Send + Sync {
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

#[derive(Clone, Default)]
pub struct SDFVolume {
    pub base: Vector3<f64>,
    pub size: Vector3<f64>,
}

impl SDFVolume {
    pub(crate) fn real_pos<const N: usize, S>(
        &self,
        norm_pos: &SVector<f64, N>,
        subspace: &S,
    ) -> SVector<f64, N>
    where
        S: Subspace<N>,
        [(); 3 - N]:,
    {
        let bp = subspace.project_vec(&self.base);
        let sp = subspace.project_vec(&self.size);
        bp + norm_pos.component_mul(&sp)
    }

    pub(crate) fn norm_pos<const N: usize, S>(
        &self,
        real_pos: &SVector<f64, N>,
        subspace: &S,
    ) -> SVector<f64, N>
    where
        S: Subspace<N>,
        [(); 3 - N]:,
    {
        let bp = subspace.project_vec(&self.base);
        let sp = subspace.project_vec(&self.size);
        (real_pos - bp).component_div(&sp)
    }
}
