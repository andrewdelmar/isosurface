use std::{mem::MaybeUninit, ops::DerefMut};

use nalgebra::Vector3;

use crate::{
    cache::EvaluationCache,
    cells::{Cell, CellEntry},
    partition::PartitionCoord,
    subspace::{R1Space, R2Space, R3Space, Subspace},
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SimplexVert<'a> {
    CellBoundary(PartitionCoord<3>),
    EdgeDual(CellEntry<'a, 1, R1Space>),
    FaceDual(CellEntry<'a, 2, R2Space>),
    VolumeDual(CellEntry<'a, 3, R3Space>),
}

impl<'a> SimplexVert<'a> {
    pub(crate) fn eval(&self, cache: &mut EvaluationCache) -> f64 {
        match self {
            SimplexVert::CellBoundary(coord) => cache.eval(coord),
            SimplexVert::EdgeDual(cell) => {
                let mut data = cell.cell_data.lock().unwrap();
                let Cell { dual_val, dual_pos } = data.deref_mut();

                *dual_val.get_or_insert_with(|| cache.eval_vec(dual_pos, &cell.subspace))
            }
            SimplexVert::FaceDual(cell) => {
                let mut data = cell.cell_data.lock().unwrap();
                let Cell { dual_val, dual_pos } = data.deref_mut();

                *dual_val.get_or_insert_with(|| cache.eval_vec(dual_pos, &cell.subspace))
            }
            SimplexVert::VolumeDual(cell) => {
                let mut data = cell.cell_data.lock().unwrap();
                let Cell { dual_val, dual_pos } = data.deref_mut();

                *dual_val.get_or_insert_with(|| cache.eval_vec(dual_pos, &cell.subspace))
            }
        }
    }

    pub(crate) fn pos(&self, cache: &mut EvaluationCache) -> Vector3<f64> {
        cache.volume.real_pos(
            &match self {
                SimplexVert::CellBoundary(coord) => coord.norm_pos(),
                SimplexVert::EdgeDual(cell) => cell
                    .subspace
                    .unproject_vec(&cell.cell_data.lock().unwrap().dual_pos),
                SimplexVert::FaceDual(cell) => cell
                    .subspace
                    .unproject_vec(&cell.cell_data.lock().unwrap().dual_pos),
                SimplexVert::VolumeDual(cell) => cell.cell_data.lock().unwrap().dual_pos,
            },
            &R3Space(),
        )
    }

    pub(crate) fn inside(&self, cache: &mut EvaluationCache) -> bool {
        self.eval(cache) < 0.0
    }
}

pub(crate) struct Simplex<'a, const N: usize> {
    pub(crate) verts: [SimplexVert<'a>; N],
}

impl<'a, const N: usize> Simplex<'a, N>
where
    SimplexVert<'a>: Sized,
{
    pub(crate) fn add(&self, vert: SimplexVert<'a>) -> Simplex<'a, { N + 1 }> {
        let verts = unsafe {
            let mut uninit = MaybeUninit::<[SimplexVert<'a>; N + 1]>::uninit();

            let ptr = uninit.as_mut_ptr() as *mut SimplexVert<'a>;
            (ptr as *mut [SimplexVert<'a>; N]).write(self.verts.clone());
            (ptr.add(N) as *mut SimplexVert).write(vert);

            uninit.assume_init()
        };

        Simplex::<{ N + 1 }> { verts }
    }
}
