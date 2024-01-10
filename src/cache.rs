use std::collections::BTreeMap;

use nalgebra::{SVector, Vector3};

use crate::{
    partition::PartitionCoord,
    subspace::{R3Space, Subspace},
    SDFVolume, VolumetricFunc,
};

// An EvaluationCache is a cache of evaluations of an SDFExpression and its gradient.
// This simplifies looking up values when constructing cell trees and during marching tetrahedra.
#[derive(Clone)]
pub(crate) struct EvaluationCache<'a> {
    func: &'a dyn VolumetricFunc,

    pub(crate) volume: &'a SDFVolume,

    func_vals: BTreeMap<PartitionCoord<3>, f64>,
    grad_vals: BTreeMap<PartitionCoord<3>, Vector3<f64>>,
}

impl<'a> EvaluationCache<'a> {
    pub(crate) fn new(func: &'a dyn VolumetricFunc, volume: &'a SDFVolume) -> Self {
        Self {
            func,
            volume,
            func_vals: BTreeMap::default(),
            grad_vals: BTreeMap::default(),
        }
    }

    pub(crate) fn eval(&mut self, at: &PartitionCoord<3>) -> f64 {
        *self.func_vals.entry(*at).or_insert_with(|| {
            self.func
                .eval(&self.volume.real_pos(&at.norm_pos(), &R3Space()))
        })
    }

    pub(crate) fn eval_vec<const N: usize, S>(
        &self,
        norm_pos: &SVector<f64, N>,
        subspace: &S,
    ) -> f64
    where
        S: Subspace<N>,
        [(); 3 - N]:,
    {
        self.func.eval(
            &self
                .volume
                .real_pos::<3, R3Space>(&subspace.unproject_vec(norm_pos), &R3Space()),
        )
    }

    pub(crate) fn eval_real(&self, real_pos: &Vector3<f64>) -> f64 {
        self.func.eval(real_pos)
    }

    pub(crate) fn eval_grad(&mut self, at: &PartitionCoord<3>) -> Vector3<f64> {
        *self.grad_vals.entry(*at).or_insert_with(|| {
            self.func
                .grad(&self.volume.real_pos(&at.norm_pos(), &R3Space()))
        })
    }
}
