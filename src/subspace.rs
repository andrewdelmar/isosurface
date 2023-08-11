use std::hash::Hash;

use nalgebra::{SVector, Vector1, Vector2, Vector3};

use crate::partition::PartitionCoord;

// A Subspace is an N-dimensional linear subspace of R3.
// Subspaces are used to make operations on collections of volume, face and edge cells more general.
pub(crate) trait Subspace<const N: usize>
where
    PartitionCoord<{ 3 - N }>:,
    Self: Clone + Sized + Hash + Ord,
{
    // Project a 3D PartitionCoord into this subspace.
    fn project_coord(&self, coord: &PartitionCoord<3>) -> PartitionCoord<N>;
    // Project a vector in R3 into this subspace.
    fn project_vec(&self, vec: &Vector3<f64>) -> SVector<f64, N>;

    // Returns a 3D  PartitionCoord corresponding to the the position of coord in this subspace. 
    fn unproject_coord(&self, coord: &PartitionCoord<N>) -> PartitionCoord<3>;
    // Returns a vector in R3 corresponding to the position of vec in this subspace.
    fn unproject_vec(&self, vec: &SVector<f64, N>) -> Vector3<f64>;

    // Returns subspaces of this type along the boundary of a volume the given PartitionCoord.
    // More specifically this will return subspaces along the faces or edges of a volume for
    // R2Space and R1Space.
    fn volume_cell_intersections(coord: &PartitionCoord<3>) -> Vec<Self>;
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) struct R3Space();

impl Subspace<3> for R3Space {
    fn project_coord(&self, coord: &PartitionCoord<3>) -> PartitionCoord<3> {
        *coord
    }

    fn project_vec(&self, vec: &Vector3<f64>) -> Vector3<f64> {
        vec.clone()
    }

    fn unproject_coord(&self, coord: &PartitionCoord<3>) -> PartitionCoord<3> {
        coord.clone()
    }

    fn unproject_vec(&self, vec: &Vector3<f64>) -> Vector3<f64> {
        *vec
    }

    fn volume_cell_intersections(_: &PartitionCoord<3>) -> Vec<Self> {
        vec![Self()]
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) enum R2Space {
    YZ(PartitionCoord<1>),
    XZ(PartitionCoord<1>),
    XY(PartitionCoord<1>),
}

impl Subspace<2> for R2Space {
    fn project_coord(&self, coord: &PartitionCoord<3>) -> PartitionCoord<2> {
        let PartitionCoord([x, y, z]) = coord;
        match self {
            R2Space::YZ(_) => PartitionCoord([*y, *z]),
            R2Space::XZ(_) => PartitionCoord([*x, *z]),
            R2Space::XY(_) => PartitionCoord([*x, *y]),
        }
    }

    fn project_vec(&self, vec: &Vector3<f64>) -> Vector2<f64> {
        match self {
            R2Space::YZ(_) => vec.yz(),
            R2Space::XZ(_) => vec.xz(),
            R2Space::XY(_) => vec.xy(),
        }
    }

    fn unproject_coord(&self, coord: &PartitionCoord<2>) -> PartitionCoord<3> {
        let PartitionCoord([u, v]) = coord.clone();
        match self {
            R2Space::YZ(PartitionCoord([x])) => PartitionCoord([*x, u, v]),
            R2Space::XZ(PartitionCoord([y])) => PartitionCoord([u, *y, v]),
            R2Space::XY(PartitionCoord([z])) => PartitionCoord([u, v, *z]),
        }
    }

    fn unproject_vec(&self, vec: &Vector2<f64>) -> Vector3<f64> {
        let (u, v) = (vec.x, vec.y);
        match self {
            R2Space::YZ(x) => Vector3::new(x.norm_pos().x, u, v),
            R2Space::XZ(y) => Vector3::new(u, y.norm_pos().x, v),
            R2Space::XY(z) => Vector3::new(u, v, z.norm_pos().x),
        }
    }

    fn volume_cell_intersections(coord: &PartitionCoord<3>) -> Vec<Self> {
        let PartitionCoord([xl, yl, zl]) = coord.low_parents();
        let PartitionCoord([xh, yh, zh]) = coord.high_parents();
        vec![
            Self::YZ(PartitionCoord([xh])),
            Self::YZ(PartitionCoord([xl])),
            Self::XZ(PartitionCoord([yl])),
            Self::XZ(PartitionCoord([yh])),
            Self::XY(PartitionCoord([zl])),
            Self::XY(PartitionCoord([zh])),
        ]
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) enum R1Space {
    X(PartitionCoord<2>),
    Y(PartitionCoord<2>),
    Z(PartitionCoord<2>),
}

impl Subspace<1> for R1Space {
    fn project_coord(&self, coord: &PartitionCoord<3>) -> PartitionCoord<1> {
        let PartitionCoord([x, y, z]) = coord;
        match self {
            R1Space::X(_) => PartitionCoord([*x]),
            R1Space::Y(_) => PartitionCoord([*y]),
            R1Space::Z(_) => PartitionCoord([*z]),
        }
    }

    fn project_vec(&self, vec: &Vector3<f64>) -> Vector1<f64> {
        match self {
            R1Space::X(_) => Vector1::new(vec.x),
            R1Space::Y(_) => Vector1::new(vec.y),
            R1Space::Z(_) => Vector1::new(vec.z),
        }
    }

    fn unproject_coord(&self, coord: &PartitionCoord<1>) -> PartitionCoord<3> {
        let PartitionCoord([c]) = coord.clone();
        match self {
            R1Space::X(PartitionCoord([y, z])) => PartitionCoord([c, *y, *z]),
            R1Space::Y(PartitionCoord([x, z])) => PartitionCoord([*x, c, *z]),
            R1Space::Z(PartitionCoord([x, y])) => PartitionCoord([*x, *y, c]),
        }
    }

    fn unproject_vec(&self, vec: &Vector1<f64>) -> Vector3<f64> {
        let c = vec.x;
        match self {
            R1Space::X(PartitionCoord([y, z])) => Vector3::new(c, y.norm_pos(), z.norm_pos()),
            R1Space::Y(PartitionCoord([x, z])) => Vector3::new(x.norm_pos(), c, z.norm_pos()),
            R1Space::Z(PartitionCoord([x, y])) => Vector3::new(x.norm_pos(), y.norm_pos(), c),
        }
    }

    fn volume_cell_intersections(coord: &PartitionCoord<3>) -> Vec<Self> {
        let PartitionCoord([xl, yl, zl]) = coord.low_parents();
        let PartitionCoord([xh, yh, zh]) = coord.high_parents();
        vec![
            Self::X(PartitionCoord([yl, zl])),
            Self::X(PartitionCoord([yh, zl])),
            Self::X(PartitionCoord([yl, zh])),
            Self::X(PartitionCoord([yh, zh])),
            Self::Y(PartitionCoord([xl, zl])),
            Self::Y(PartitionCoord([xh, zl])),
            Self::Y(PartitionCoord([xl, zh])),
            Self::Y(PartitionCoord([xh, zh])),
            Self::Z(PartitionCoord([xl, yl])),
            Self::Z(PartitionCoord([xh, yl])),
            Self::Z(PartitionCoord([xl, yh])),
            Self::Z(PartitionCoord([xh, yh])),
        ]
    }
}

impl R2Space {
    pub(crate) fn edges(&self, coord: &PartitionCoord<2>) -> [(PartitionCoord<1>, R1Space); 4] {
        match self {
            R2Space::YZ(PartitionCoord([x])) => {
                let PartitionCoord([y, z]) = *coord;
                let PartitionCoord([yl, zl]) = coord.low_parents();
                let PartitionCoord([yh, zh]) = coord.high_parents();
                [
                    (PartitionCoord([y]), R1Space::Y(PartitionCoord([*x, zl]))),
                    (PartitionCoord([y]), R1Space::Y(PartitionCoord([*x, zh]))),
                    (PartitionCoord([z]), R1Space::Z(PartitionCoord([*x, yl]))),
                    (PartitionCoord([z]), R1Space::Z(PartitionCoord([*x, yh]))),
                ]
            }
            R2Space::XZ(PartitionCoord([y])) => {
                let PartitionCoord([x, z]) = *coord;
                let PartitionCoord([xl, zl]) = coord.low_parents();
                let PartitionCoord([xh, zh]) = coord.high_parents();
                [
                    (PartitionCoord([x]), R1Space::X(PartitionCoord([*y, zl]))),
                    (PartitionCoord([x]), R1Space::X(PartitionCoord([*y, zh]))),
                    (PartitionCoord([z]), R1Space::Z(PartitionCoord([xl, *y]))),
                    (PartitionCoord([z]), R1Space::Z(PartitionCoord([xh, *y]))),
                ]
            }
            R2Space::XY(PartitionCoord([z])) => {
                let PartitionCoord([x, y]) = *coord;
                let PartitionCoord([xl, yl]) = coord.low_parents();
                let PartitionCoord([xh, yh]) = coord.high_parents();
                [
                    (PartitionCoord([x]), R1Space::X(PartitionCoord([yl, *z]))),
                    (PartitionCoord([x]), R1Space::X(PartitionCoord([yh, *z]))),
                    (PartitionCoord([y]), R1Space::Y(PartitionCoord([xl, *z]))),
                    (PartitionCoord([y]), R1Space::Y(PartitionCoord([xh, *z]))),
                ]
            }
        }
    }
}
