use std::collections::BTreeMap;

use nalgebra::Vector3;

use crate::{
    isosurface::SDFVolume,
    partition::PartitionCoord,
    sdf::{Dimension, SDFExpression},
};

pub(crate) struct EvaluationCache<'a> {
    f: &'a SDFExpression,
    df_dx: SDFExpression,
    df_dy: SDFExpression,
    df_dz: SDFExpression,

    volume: &'a SDFVolume,

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

    pub(crate) fn pos(&mut self, at: &PartitionCoord<3>) -> Vector3<f64> {
        self.volume.point_pos(&at.pos())
    }

    pub(crate) fn eval(&mut self, at: &PartitionCoord<3>) -> f64 {
        *self
            .f_vals
            .entry(*at)
            .or_insert_with(|| self.f.eval(&self.volume.point_pos(&at.pos())))
    }

    pub(crate) fn eval_grad(&mut self, at: &PartitionCoord<3>) -> Vector3<f64> {
        let x = *self
            .df_dx_vals
            .entry(*at)
            .or_insert_with(|| self.df_dx.eval(&self.volume.point_pos(&at.pos())));

        let y = *self
            .df_dy_vals
            .entry(*at)
            .or_insert_with(|| self.df_dy.eval(&self.volume.point_pos(&at.pos())));

        let z = *self
            .df_dz_vals
            .entry(*at)
            .or_insert_with(|| self.df_dz.eval(&self.volume.point_pos(&at.pos())));

        Vector3::new(x, y, z)
    }
}
