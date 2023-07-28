use std::collections::{btree_map, BTreeMap};

use crate::{
    partition::{PartitionCoord, PartitionTreeIter},
    subspace::Subspace,
};

use super::{
    collection::{Cell, CellTree},
    CellCollection,
};

#[derive(Clone)]
pub(crate) struct CellView<'a, const N: usize, S>
where
    [(); 3 - N]:,
    S: Subspace<N>,
{
    pub(crate) subspace: S,
    pub(crate) coord: PartitionCoord<N>,
    pub(crate) cell_data: &'a Cell<N>,
}

pub(crate) struct CellIter<'a, const N: usize, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    b_iter: btree_map::Iter<'a, S, CellTree<N>>,
    leaves: Option<(&'a S, PartitionTreeIter<'a, Cell<N>, N>)>,
}

impl<'a, const N: usize, S> CellIter<'a, N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    fn new(b_tree: &'a BTreeMap<S, CellTree<N>>) -> Self {
        let mut b_iter = b_tree.iter();
        let leaves = match b_iter.next() {
            Some((subspace, tree)) => Some((subspace, tree.into_iter())),
            None => None,
        };
        Self { b_iter, leaves }
    }
}

impl<'a, const N: usize, S> Iterator for CellIter<'a, N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    type Item = CellView<'a, N, S>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.leaves {
            Some((subspace, ref mut leaf_iter)) => match leaf_iter.next() {
                Some((coord, cell_data)) => Some(CellView {
                    subspace: subspace.clone(),
                    coord,
                    cell_data,
                }),
                None => {
                    self.leaves = match self.b_iter.next() {
                        Some((subspace, tree)) => Some((subspace, tree.into_iter())),
                        None => None,
                    };
                    self.next()
                }
            },
            None => None,
        }
    }
}

impl<'a, const N: usize, S> IntoIterator for &'a CellCollection<N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    type Item = CellView<'a, N, S>;

    type IntoIter = CellIter<'a, N, S>;

    fn into_iter(self) -> Self::IntoIter {
        CellIter::new(&self.0)
    }
}


pub(crate) struct ChildIterator<'a, const N: usize, S>
where
    [(); 1 << N]:,
{
    subspace: S,
    tree_iter: Option<PartitionTreeIter<'a, Cell<N>, N>>,
}

impl<'a, const N: usize, S> Iterator for ChildIterator<'a, N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    type Item = CellView<'a, N, S>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tree_iter) = &mut self.tree_iter 
        && let Some((coord, cell)) = tree_iter.next() {
            Some(CellView {
                subspace: self.subspace.clone(),
                coord: coord,
                cell_data: cell,
            })
        } else {
            None
        }
    }
}

impl<const N: usize, S> CellCollection<N, S>
where
    [(); 3 - N]:,
    [(); 1 << N]:,
    S: Subspace<N>,
{
    pub(super) fn children<'a>(
        &'a self,
        coord: &PartitionCoord<N>,
        subspace: &S,
    ) -> ChildIterator<'a, N, S> {
        let Self(tree) = self;

        let tree_iter = tree.get(subspace).map(|p_tree| p_tree.children(coord));

        ChildIterator {
            subspace: subspace.clone(),
            tree_iter,
        }
    }
}
