use nalgebra::{SMatrix, SVector};

use crate::{cache::EvaluationCache, partition::PartitionCoord, subspace::Subspace};

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
                let real_pos = cache.volume.real_pos(&vert_coord.norm_pos(), subspace);

                let coord3 = subspace.unproject_coord(&vert_coord);
                let grad3 = cache.eval_grad(&coord3).normalize();
                let grad_s = subspace.project_vec(&grad3);

                let d = -grad_s.dot(&real_pos);
                let plane = grad_s.push(d);

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
                //TODO This should project the plane equations onto the cell
                // to guarantee a position in the volume
                coord.norm_pos()
            }
        }
    };
}

impl_optimizer!(1, find_edge_dual);
impl_optimizer!(2, find_face_dual);
impl_optimizer!(3, find_volume_dual);
