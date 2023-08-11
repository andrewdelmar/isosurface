use crate::partition::PartitionCoord;

// A partition tree represents spatial partitions in N dimensional space
// (an octree in 3D, quadtree in 2D or binary tree in 1D).
//
// Values in a partition tree are only stored at the leaves.
// If a value is added at a node below an existing one with a value, the existing valus is removed.
pub(crate) enum PartitionTree<T, const N: usize>
where
    [(); 1 << N]:,
{
    None,
    Node(Box<[PartitionTree<T, N>; 1 << N]>),
    Leaf(T),
}

impl<T, const N: usize> PartitionTree<T, N>
where
    [(); 1 << N]:,
{
    // Insert a node at the given coordinate, if a child of that coordinate doesn't already exist.
    // It will remove the values of parents of this coordinate.
    pub(crate) fn insert_leaf(&mut self, coord: PartitionCoord<N>, val: T) {
        match (coord.is_root(), &self) {
            (false, PartitionTree::None) | (false, PartitionTree::Leaf(_)) => {
                *self = Self::Node(Box::new(core::array::from_fn(|_| Self::None)));
                self.insert_leaf(coord, val);
            }
            (false, PartitionTree::Node(_)) => {
                if let PartitionTree::Node(c) = self {
                    c[coord.tree_index()].insert_leaf(coord.id_at_child(), val)
                }
            }
            (true, PartitionTree::None) | (true, PartitionTree::Leaf(_)) => *self = Self::Leaf(val),
            (true, PartitionTree::Node(_)) => {}
        };
    }

    // Remove any nodes without children.
    pub(crate) fn prune(&mut self) {
        if let Self::Node(box children) = self {
            let mut empty = true;
            for child in children {
                child.prune();
                match child {
                    PartitionTree::None => {}
                    PartitionTree::Node(_) | PartitionTree::Leaf(_) => empty = false,
                }
            }

            if empty {
                *self = Self::None;
            }
        }
    }

    // Returns an iterator of all children of the given coordinate.
    pub(crate) fn children<'a>(&'a self, coord: &PartitionCoord<N>) -> PartitionTreeIter<'a, T, N>
    where
        [(); 1 << N]:,
    {
        self.children_with_coord(coord, coord)
    }

    fn children_with_coord<'a>(
        &'a self,
        coord: &PartitionCoord<N>,
        original_coord: &PartitionCoord<N>,
    ) -> PartitionTreeIter<'a, T, N>
    where
        [(); 1 << N]:,
    {
        match (coord.is_root(), self) {
            (true, node) => PartitionTreeIter {
                stack: vec![(node, *original_coord)],
            },
            (false, PartitionTree::None | PartitionTree::Leaf(_)) => PartitionTreeIter {
                stack: Vec::default(),
            },
            (false, PartitionTree::Node(c)) => {
                c[coord.tree_index()].children_with_coord(&coord.id_at_child(), original_coord)
            }
        }
    }
}

impl<T, const N: usize> Default for PartitionTree<T, N>
where
    [(); 1 << N]:,
{
    fn default() -> Self {
        Self::None
    }
}

pub(crate) struct PartitionTreeIter<'a, T, const N: usize>
where
    [(); 1 << N]:,
{
    stack: Vec<(&'a PartitionTree<T, N>, PartitionCoord<N>)>,
}

impl<'a, T, const N: usize> Iterator for PartitionTreeIter<'a, T, N>
where
    [(); 1 << N]:,
{
    type Item = (PartitionCoord<N>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            Some((t, coord)) => match t {
                PartitionTree::None => self.next(),
                PartitionTree::Node(box children) => {
                    for child in std::iter::zip(children, coord.child_coords()).rev() {
                        self.stack.push(child);
                    }
                    self.next()
                }
                PartitionTree::Leaf(val) => Some((coord, val)),
            },
            None => None,
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a PartitionTree<T, N>
where
    [(); 1 << N]:,
{
    type Item = (PartitionCoord<N>, &'a T);

    type IntoIter = PartitionTreeIter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        PartitionTreeIter {
            stack: vec![(self, PartitionCoord::default())],
        }
    }
}
