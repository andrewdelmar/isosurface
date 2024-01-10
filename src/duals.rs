use std::{collections::BTreeSet, thread::scope};

use crossbeam_channel::{unbounded, Receiver};
use nalgebra::{SMatrix, SVector};

use crate::{
    cache::EvaluationCache,
    cells::{CellCollection, CellEntry},
    partition::PartitionCoord,
    subspace::Subspace,
};

// Because of the ridiculously painful bounds required for pseudo_inverse and other matrix operations
// it's much shorter to just implement this for N = 1, 2, and 3 with a macro than use generics.
macro_rules! impl_find {
    ($Dim: literal, $Func: ident ) => {
        fn $Func<S: Subspace<$Dim>>(
            coord: &PartitionCoord<$Dim>,
            subspace: &S,
            cache: &mut EvaluationCache,
            pow: usize,
        ) -> SVector<f64, $Dim> {
            let mut quadric = SMatrix::<f64, { $Dim + 1 }, { $Dim + 1 }>::default();
            for vert_coord in subdivide_coord(coord, pow) {
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

fn subdivide_coord<const N: usize>(coord: &PartitionCoord<N>, pow: usize) -> Vec<PartitionCoord<N>>
where
    [(); 1 << N]:,
{
    let mut coords = BTreeSet::new();
    coords.insert(coord.clone());

    for _ in 0..pow {
        coords = coords
            .iter()
            .flat_map(PartitionCoord::<N>::child_coords)
            .collect();
    }

    coords.into_iter().collect()
}

impl_find!(1, find_edge_dual);
impl_find!(2, find_face_dual);
impl_find!(3, find_volume_dual);

macro_rules! impl_worker {
    ($Dim: literal, $FindFunc: ident, $Func: ident) => {
        fn $Func<'a, S: Subspace<$Dim>>(
            tasks: Receiver<Vec<CellEntry<'a, $Dim, S>>>,
            mut cache: EvaluationCache,
            pow: usize,
        ) {
            for task in tasks.iter() {
                for cell in task {
                    cell.cell_data.lock().unwrap().dual_pos =
                        $FindFunc(&cell.coord, &cell.subspace, &mut cache, pow);
                }
            }
        }
    };
}

impl_worker!(1, find_edge_dual, find_edge_dual_worker);
impl_worker!(2, find_face_dual, find_face_dual_worker);
impl_worker!(3, find_volume_dual, find_volume_dual_worker);

const TASK_SIZE: usize = 1000;

macro_rules! impl_find_all {
    ($Dim: literal, $WorkerFunc: ident, $FindFunc: ident, $Func: ident) => {
        pub(crate) fn $Func<S: Subspace<$Dim> + Send>(
            cells: &CellCollection<$Dim, S>,
            cache: &mut EvaluationCache,
            worker_threads: usize,
            subdivisions: usize,
        ) {
            if worker_threads > 0 {
                scope(|s| {
                    let (task_s, task_r) = unbounded();

                    for _ in 0..worker_threads {
                        let task_r = task_r.clone();
                        let cache = cache.clone();
                        s.spawn(move || $WorkerFunc(task_r, cache, subdivisions));
                    }

                    let mut task = Vec::with_capacity(TASK_SIZE);
                    for cell in cells {
                        task.push(cell);

                        if task.len() == TASK_SIZE {
                            task_s.send(task).expect("Failed to send volume cell.");
                            task = Vec::with_capacity(TASK_SIZE);
                        }
                    }

                    if !task.is_empty() {
                        task_s.send(task).expect("Failed to send volume cell.");
                    }
                });
            } else {
                for cell in cells {
                    cell.cell_data.lock().unwrap().dual_pos =
                        $FindFunc(&cell.coord, &cell.subspace, cache, subdivisions);
                }
            }
        }
    };
}

impl_find_all!(
    3,
    find_volume_dual_worker,
    find_volume_dual,
    find_all_volume_duals
);
impl_find_all!(
    2,
    find_face_dual_worker,
    find_face_dual,
    find_all_face_duals
);
impl_find_all!(
    1,
    find_edge_dual_worker,
    find_edge_dual,
    find_all_edge_duals
);
