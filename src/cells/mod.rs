use crate::subspace::{R1Space, R2Space, R3Space};

mod collection;
pub(crate) use collection::CellCollection;

pub(crate) type EdgeCellCollection = CellCollection<1, R1Space>;
pub(crate) type FaceCellCollection = CellCollection<2, R2Space>;
pub(crate) type VolumeCellCollection = CellCollection<3, R3Space>;

mod iterator;
pub(crate) use iterator::CellEntry;

mod build;
pub(crate) use build::build_cell_trees;

mod tetrahedralize;
pub(crate) use tetrahedralize::tetrahedralize;
