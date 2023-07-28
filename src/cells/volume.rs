use std::collections::BTreeMap;

use nalgebra::{SVector, Vector3};

use crate::{
    cache::EvaluationCache, partition::PartitionTree, partition::PartitionCoord,
    subspace::R3Space,
};

use super::VolumeCellTree;

impl VolumeCellTree {
    pub(crate) fn new(cache: &mut EvaluationCache, min_depth: usize, max_depth: usize) -> Self {
        let mut tree =
            Self::tree_with_min_depth(cache, min_depth, max_depth, PartitionCoord::default());
        tree.prune();
        let mut out = Self(BTreeMap::default());
        out.0.insert(R3Space(), tree);
        out
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

    fn tree_with_min_depth(
        cache: &mut EvaluationCache,
        min_depth: usize,
        max_depth: usize,
        coord: PartitionCoord<3>,
    ) -> CellTree<3> {
        let children = match min_depth {
            0 => coord
                .child_coords()
                .map(|c| Self::tree(cache, max_depth, c)),
            _ => coord
                .child_coords()
                .map(|c| Self::tree_with_min_depth(cache, min_depth - 1, max_depth - 1, c)),
        };

        let out = VolumeLeafTree::Node(Box::new(children));

        out
    }

    fn tree(
        cache: &mut EvaluationCache,
        max_depth: usize,
        coord: PartitionCoord<3>,
    ) -> VolumeLeafTree {
        let sign_change = Self::sign_change(cache, coord);
        match (max_depth, sign_change) {
            (0, true) => VolumeLeafTree::Leaf(Vector3::default()),
            (0, false) => VolumeLeafTree::None,
            (_, true) => VolumeLeafTree::Node(Box::new(
                coord
                    .child_coords()
                    .map(|c| Self::tree(cache, max_depth - 1, c)),
            )),
            (_, false) => VolumeLeafTree::None,
        }
    }
}
