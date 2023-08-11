use std::collections::BTreeMap;

use crate::{
    cache::EvaluationCache,
    cells::CellCollection,
    partition::{PartitionCoord, PartitionTree},
    subspace::R3Space,
};

use super::{
    collection::{Cell, CellTree},
    EdgeCellCollection, FaceCellCollection, VolumeCellCollection,
};

// build_cell_trees returns an octree, a set of quadtrees and a set of binary trees.
// These sets of trees contain volume, face and edge cells respectively.
// Trees are divided wherever a sign change occurs in the function up to max_depth.
// Trees of cells are divided up to min_depth times before sign changes are tested.   
pub(crate) fn build_cell_trees(
    cache: &mut EvaluationCache,
    min_depth: usize,
    max_depth: usize,
) -> (VolumeCellCollection, FaceCellCollection, EdgeCellCollection) {
    let mut volume_tree =
        volume_tree_with_min_depth(cache, min_depth, max_depth, PartitionCoord::default());
    volume_tree.prune();
    let mut volume_b_tree = BTreeMap::<R3Space, CellTree<3>>::default();
    volume_b_tree.insert(R3Space(), volume_tree);

    let volume_cells = CellCollection::<3, R3Space>(volume_b_tree);
    let face_cells = FaceCellCollection::build_from_volume_cells(&volume_cells);
    let edge_cells = EdgeCellCollection::build_from_volume_cells(&volume_cells);

    (volume_cells, face_cells, edge_cells)
}

fn sign_change(cache: &mut EvaluationCache, coord: PartitionCoord<3>) -> bool {
    let children = coord.vertex_coords();
    let sign = cache.eval(&children[0]) > 0.0;
    for c in coord.vertex_coords() {
        if (cache.eval(&c) > 0.0) != sign {
            return true;
        }
    }

    false
}

fn volume_tree_with_min_depth(
    cache: &mut EvaluationCache,
    min_depth: usize,
    max_depth: usize,
    coord: PartitionCoord<3>,
) -> CellTree<3> {
    let children = match min_depth {
        0 => coord
            .child_coords()
            .map(|c| volume_tree(cache, max_depth, c)),
        _ => coord
            .child_coords()
            .map(|c| volume_tree_with_min_depth(cache, min_depth - 1, max_depth - 1, c)),
    };

    let out = PartitionTree::Node(Box::new(children));

    out
}

fn volume_tree(
    cache: &mut EvaluationCache,
    max_depth: usize,
    coord: PartitionCoord<3>,
) -> CellTree<3> {
    let sign_change = sign_change(cache, coord);
    match (max_depth, sign_change) {
        (0, true) => PartitionTree::Leaf(Cell::<3>::default()),
        (0, false) => PartitionTree::None,
        (_, true) => PartitionTree::Node(Box::new(
            coord
                .child_coords()
                .map(|c| volume_tree(cache, max_depth - 1, c)),
        )),
        (_, false) => PartitionTree::None,
    }
}
