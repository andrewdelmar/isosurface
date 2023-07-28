use std::{
    cell::{OnceCell, RefCell},
    collections::BTreeMap,
};

use nalgebra::SVector;

use crate::{partition::PartitionTree, subspace::Subspace};

use super::VolumeCellCollection;

pub(crate) struct Cell<const N: usize> {
    pub(crate) dual_pos: RefCell<SVector<f64, N>>,
    pub(crate) vertex_buffer_ind: OnceCell<usize>,
}

impl<const N: usize> Default for Cell<N> {
    fn default() -> Self {
        Self {
            dual_pos: RefCell::new(SVector::<f64, N>::from_element(Default::default())),
            vertex_buffer_ind: Default::default(),
        }
    }
}

pub(super) type CellTree<const N: usize> = PartitionTree<Cell<N>, N>;

pub(crate) struct CellCollection<const N: usize, S>(pub(super) BTreeMap<S, CellTree<N>>)
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>;

impl<const N: usize, S> CellCollection<N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    pub(super) fn build_from_volume_cells(volume_cells: &VolumeCellCollection) -> Self {
        let mut tree = BTreeMap::<S, CellTree<N>>::default();

        for cell in volume_cells {
            for subspace in S::volume_cell_intersections(&cell.coord) {
                let proj = subspace.project_coord(&cell.coord);
                tree.entry(subspace)
                    .or_default()
                    .insert_leaf(proj, Cell::<N>::default())
            }
        }

        Self(tree)
    }
}
