use std::fmt::Display;

use nalgebra::SVector;

use crate::partition::PartitionID;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct PartitionCoord<const N: usize>(pub(crate) [PartitionID; N]);

impl<const N: usize> PartitionCoord<N> {
    pub(crate) fn tree_index(&self) -> usize {
        let Self(coords) = self;
        let mut index = 0;
        let mut mul = 1;
        for i in coords {
            index += i.tree_index() * mul;
            mul <<= 1;
        }
        index
    }

    pub(crate) fn id_at_child(&self) -> Self {
        let Self(coords) = self;
        Self(coords.map(|c| c.id_at_child()))
    }

    pub(crate) fn is_root(&self) -> bool {
        let Self(coords) = self;
        for c in coords {
            if c.is_root() {
                return true;
            }
        }
        false
    }

    pub(crate) fn child_coords(&self) -> [PartitionCoord<N>; 1 << N] {
        let mut coords = [*self; 1 << N];
        for index in 0..1 << N {
            let mut index_bits = index;
            for dim in 0..N {
                coords[index].0[dim] = if index_bits & 0b1 != 0 {
                    coords[index].0[dim].high_child()
                } else {
                    coords[index].0[dim].low_child()
                };

                index_bits >>= 1;
            }
        }
        coords
    }

    pub(crate) fn high_parents(&self) -> Self {
        let Self(coords) = self;
        Self(coords.map(|c| c.high_parent()))
    }

    pub(crate) fn low_parents(&self) -> Self {
        let Self(coords) = self;
        Self(coords.map(|c| c.low_parent()))
    }

    pub(crate) fn vertex_coords(&self) -> [PartitionCoord<N>; 1 << N] {
        let mut coords = [self.clone(); 1 << N];

        for index in 0..(1 << N) {
            let mut index_div = index;
            for dim in 0..N {
                if index_div % 2 == 0 {
                    coords[index].0[dim] = coords[index].0[dim].low_parent()
                } else {
                    coords[index].0[dim] = coords[index].0[dim].high_parent()
                }
                index_div /= 2;
            }
        }

        coords
    }

    pub(crate) fn norm_pos(&self) -> SVector<f64, N> {
        SVector::<f64, N>::from_iterator(self.0.map(|p| p.pos()))
    }
}

impl<const N: usize> Default for PartitionCoord<N> {
    fn default() -> Self {
        Self([Default::default(); N])
    }
}

impl<const N: usize> Display for PartitionCoord<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut first = true;
        for c in self.0 {
            if !first {
                write!(f, ", ")?;
            } else {
                first = false;
            }
            c.fmt(f)?
        }
        write!(f, "]")
    }
}
