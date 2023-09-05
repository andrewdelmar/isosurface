use std::ops;

use nalgebra::{
    allocator::Allocator, Const, DefaultAllocator, DimAdd, DimSum, SMatrix, SVector, U1,
};

use crate::{
    cache::EvaluationCache,
    partition::PartitionCoord,
    subspace::{R3Space, Subspace},
};

// Because of the ridiculously painful bounds required for pseudo_inverse and other matrix operations
// it's much shorter to just implement this for N = 1, 2, and 3 with a macro than use generics.
macro_rules! impl_optimizer {
    ($Dim: literal, $Func: ident ) => {
        pub(crate) fn $Func<S: Subspace<$Dim>>(
            coord: &PartitionCoord<$Dim>,
            subspace: &S,
            cache: &mut EvaluationCache,
        ) -> SVector<f64, $Dim> {
            let mut quadric = SMatrix::<f64, { $Dim + 1 }, { $Dim + 1 }>::default();
            for vert_coord in coord.vertex_coords() {
                let coord3 = subspace.unproject_coord(&vert_coord);

                let real_pos3 = cache.volume.real_pos(&coord3.norm_pos(), &R3Space());
                let real_pos_o = subspace.ortho_components_vec(&real_pos3);

                let grad3 = cache.eval_grad(&coord3).normalize();
                let grad_s = subspace.project_vec(&grad3);
                let grad_o = subspace.ortho_components_vec(&grad3);

                let val = cache.eval(&coord3);

                let d = val - grad3.dot(&real_pos3);
                let d_s = d + grad_o.dot(&real_pos_o);

                let plane = grad_s.push(d_s);

                quadric += plane * plane.transpose();
            }

            let a = quadric.fixed_view::<$Dim, $Dim>(0, 0);
            let b = quadric.fixed_view::<$Dim, 1>(0, $Dim);
            let i = a.pseudo_inverse(f64::EPSILON).unwrap();
            let dual_pos = i * (-b);
            let norm_pos = cache.volume.norm_pos(&dual_pos, subspace);

            if coord.inside(&norm_pos) {
                norm_pos
            } else {
                //TODO This should project the plance equations onto the cell 
                // to guarantee a position in the volume
                coord.norm_pos()
            }
        }
    };
}

impl_optimizer!(1, find_edge_dual);
impl_optimizer!(2, find_face_dual);
impl_optimizer!(3, find_volume_dual);
