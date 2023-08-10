use nalgebra::{SVector, Vector3};

use crate::subspace::Subspace;

#[derive(Clone, Default)]
pub struct SDFVolume {
    pub(crate) base: Vector3<f64>,
    pub(crate) size: Vector3<f64>,
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
