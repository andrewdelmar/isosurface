use std::fmt::Display;

// A PartitionID represents a point in space from 0 to 1. It is effectively a 63 bit fixed point value.
// It's also used used to identify a node in a binary division of space with that point at its center.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub(crate) struct PartitionID(u64);

impl PartitionID {
    const MAX_ID: u64 = 1 << 63;
    const ROOT_ID: u64 = 1 << 62;
    const TREE_BITS: u64 = !(1 << 63);

    // The upper bound of a segment represented by this ID.
    // A segment with that bound at its center is always a parent of this segment. 
    pub(crate) fn high_parent(&self) -> Self {
        let Self(id) = self;
        Self(id + (1 << (id.trailing_zeros())))
    }

    // The lower bound of a segment represented by this ID.
    // A segment with that bound at its center is always a parent of this segment. 
    pub(crate) fn low_parent(&self) -> Self {
        let Self(id) = self;
        Self(id - (1 << (id.trailing_zeros())))
    }

    // The upper half of this segment.
    // A segment of (0 - 0.5) would have a higher child of (0.25 - 0.5).
    // It is always a direct child of this segment.
    pub(crate) fn high_child(&self) -> Self {
        let Self(id) = self;
        Self(id + (1 << (id.trailing_zeros() - 1)))
    }

    // The lower half of this segment.
    // A segment of (0 - 0.5) would have a lower child of (0 - 0.25).
    // It is always a direct child of this segment.
    pub(crate) fn low_child(&self) -> Self {
        let Self(id) = self;
        Self(id - (1 << (id.trailing_zeros() - 1)))
    }

    // The position of this segment as a float from 0.0 to 1.0
    pub(crate) fn norm_pos(&self) -> f64 {
        let Self(id) = self;
        if *id == 0 {
            return 0.0;
        }

        let num = id >> id.trailing_zeros();
        let den = Self::MAX_ID >> id.trailing_zeros();

        num as f64 / den as f64
    }

    // The index in an array of children in the root node of a tree that points toward this ID.
    pub fn tree_index(&self) -> usize {
        let Self(id) = self;
        if id & Self::ROOT_ID == 0 {
            0
        } else {
            1
        }
    }

    // The equivalent ID if we treat a child, indexed by tree_index as root.
    pub(crate) fn id_at_child(&self) -> Self {
        let Self(id) = self;
        Self((id << 1) & Self::TREE_BITS)
    }

    pub(crate) fn is_root(&self) -> bool {
        let Self(id) = self;
        *id == Self::ROOT_ID
    }
}

impl Default for PartitionID {
    fn default() -> Self {
        Self(Self::ROOT_ID)
    }
}

impl Display for PartitionID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write! {f, "({} - {})", self.low_parent().norm_pos(), self.high_parent().norm_pos()}
    }
}


#[cfg(test)]
mod tests {
    use super::PartitionID;

    #[test]
    fn parent_is_inverse_of_child() {
        let mut id = PartitionID::default();
        let root = PartitionID::default();

        for _ in 0..60 {
            id = id.low_child();
        }

        for _ in 0..60 {
            id = id.high_parent();
        }

        assert_eq!(id, root);
    }

    #[test]
    fn partition_position() {
        let mut id = PartitionID::default();
        let mut expected_pos = 0.5;

        for _ in 0..60 {
            assert_eq!(id.norm_pos(), expected_pos);

            id = id.low_child();
            expected_pos /= 2.0;
        }
    }

    #[test]
    fn id_at_child_reaches_root() {
        let mut id = PartitionID::default();
        let root = PartitionID::default();

        for _ in 0..20 {
            id = id.high_child();
        }

        for _ in 0..20 {
            assert!(!id.is_root());
            id = id.id_at_child();
        }

        assert!(id.is_root());
        assert_eq!(id, root);
    }

    #[test]
    fn max_min_id_pos() {
        let root = PartitionID::default();

        let max = root.high_parent();
        assert_eq!(max.norm_pos(), 1.0);

        let min = root.low_parent();
        assert_eq!(min.norm_pos(), 0.0);
    }
}