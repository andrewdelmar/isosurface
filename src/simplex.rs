use std::mem::MaybeUninit;

use crate::{
    cells::CellView,
    partition::PartitionCoord,
    subspace::{R1Space, R2Space, R3Space},
};

#[derive(Clone)]
pub(crate) enum Vertex<'a> {
    CellBoundary(PartitionCoord<3>),
    EdgeDual(CellView<'a, 1, R1Space>),
    FaceDual(CellView<'a, 2, R2Space>),
    VolumeDual(CellView<'a, 3, R3Space>),
}

pub(crate) struct Simplex<'a, const N: usize> {
    pub(crate) verts: [Vertex<'a>; N],
}

impl<'a, const N: usize> Simplex<'a, N>
where
    Vertex<'a>: Sized,
{
    pub(crate) fn add(&self, vert: Vertex<'a>) -> Simplex<'a, { N + 1 }> {
        let verts = unsafe {
            let mut uninit = MaybeUninit::<[Vertex<'a>; N + 1]>::uninit();

            let ptr = uninit.as_mut_ptr() as *mut Vertex<'a>;
            (ptr as *mut [Vertex<'a>; N]).write(self.verts.clone());
            (ptr.add(N) as *mut Vertex).write(vert);

            uninit.assume_init()
        };

        Simplex::<{ N + 1 }> { verts }
    }
}
