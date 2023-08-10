use nalgebra::{SMatrix, SVector};

use crate::{
    cache::EvaluationCache,
    partition::PartitionCoord,
    subspace::{R3Space, Subspace},
};

#[derive(Default)]
pub(crate) struct DualQuadric<const N: usize>(SMatrix<f64, { N + 1 }, { N + 1 }>)
where
    [[f64; N + 1]; N + 1]: Default;

// Because of the painful bounds required for pseudo_inverse
// it's much shorter to just implement this for N = 1, 2, and 3.
macro_rules! impl_quadric {
    ($D: literal ) => {
        impl DualQuadric<$D> {
            pub(crate) fn add<S: Subspace<$D>>(
                &mut self,
                coord: &PartitionCoord<$D>,
                subspace: &S,
                cache: &mut EvaluationCache,
            ) {
                let coord3 = subspace.unproject_coord(coord);
                let real_pos =
                    subspace.project_vec(&cache.volume.real_pos(&coord3.norm_pos(), &R3Space()));
                let grad = subspace.project_vec(&cache.eval_grad(&coord3));
                let val = cache.eval(&coord3);

                let d = val - grad.dot(&real_pos);
                let plane = grad.push(d);

                self.0 += plane * plane.transpose();
            }

            pub(crate) fn solve(&self) -> Option<SVector<f64, $D>> {
                let a = self.0.fixed_view::<$D, $D>(0, 0);
                let b = self.0.fixed_view::<$D, 1>(0, $D);
                if let Ok(i) = a.pseudo_inverse(f64::EPSILON) {
                    Some(i * (-b))
                } else {
                    None
                }
            }

            pub(crate) fn find_dual<S: Subspace<$D>>(
                coord: &PartitionCoord<$D>,
                subspace: &S,
                cache: &mut EvaluationCache,
            ) -> SVector<f64, $D> {
                let mut o = Self::default();

                for v in coord.vertex_coords() {
                    o.add(&v, subspace, cache);
                }

                o.solve()
                    .map(|p| cache.volume.norm_pos(&p, subspace))
                    .unwrap_or(coord.norm_pos())
            }
        }
    };
}

impl_quadric!(1);
impl_quadric!(2);
impl_quadric!(3);
