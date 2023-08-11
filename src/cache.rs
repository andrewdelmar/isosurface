use std::collections::BTreeMap;

use nalgebra::{SVector, Vector3};

use crate::{
    partition::PartitionCoord,
    sdf::{Dimension, SDFExpression},
    subspace::{R3Space, Subspace},
    volume::SDFVolume,
};

// An EvaluationCache is a cache of evaluations of an SDFExpression and its gradient.
// This simplifies looking up values when constructing cell trees and during marching tetrahedra.
pub(crate) struct EvaluationCache<'a> {
    f: &'a SDFExpression,
    df_dx: SDFExpression,
    df_dy: SDFExpression,
    df_dz: SDFExpression,

    pub(crate) volume: &'a SDFVolume,

    f_vals: BTreeMap<PartitionCoord<3>, f64>,
    df_dx_vals: BTreeMap<PartitionCoord<3>, f64>,
    df_dy_vals: BTreeMap<PartitionCoord<3>, f64>,
    df_dz_vals: BTreeMap<PartitionCoord<3>, f64>,
}

impl<'a> EvaluationCache<'a> {
    pub(crate) fn new(expr: &'a SDFExpression, volume: &'a SDFVolume) -> Self {
        Self {
            f: expr,
            df_dx: expr.derive(&Dimension::X),
            df_dy: expr.derive(&Dimension::Y),
            df_dz: expr.derive(&Dimension::Z),
            volume,
            f_vals: BTreeMap::default(),
            df_dx_vals: BTreeMap::default(),
            df_dy_vals: BTreeMap::default(),
            df_dz_vals: BTreeMap::default(),
        }
    }

    pub(crate) fn eval(&mut self, at: &PartitionCoord<3>) -> f64 {
        *self.f_vals.entry(*at).or_insert_with(|| {
            self.f
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
        self.f.eval(
            &self
                .volume
                .real_pos::<3, R3Space>(&subspace.unproject_vec(norm_pos), &R3Space()),
        )
    }

    pub(crate) fn eval_grad(&mut self, at: &PartitionCoord<3>) -> Vector3<f64> {
        let x = *self.df_dx_vals.entry(*at).or_insert_with(|| {
            self.df_dx
                .eval(&self.volume.real_pos(&at.norm_pos(), &R3Space()))
        });

        let y = *self.df_dy_vals.entry(*at).or_insert_with(|| {
            self.df_dy
                .eval(&self.volume.real_pos(&at.norm_pos(), &R3Space()))
        });

        let z = *self.df_dz_vals.entry(*at).or_insert_with(|| {
            self.df_dz
                .eval(&self.volume.real_pos(&at.norm_pos(), &R3Space()))
        });

        Vector3::new(x, y, z)
    }
}
