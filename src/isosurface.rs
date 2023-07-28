use nalgebra::Vector3;

use crate::{
    cache::EvaluationCache,
    cells::{build_cell_trees, tetrahedralize},
    optimizer::DualQuadric,
    sdf::SDFExpression,
};

#[derive(Clone, Default)]
pub struct SDFVolume {
    pub(crate) base: Vector3<f64>,
    pub(crate) size: f64,
}

impl SDFVolume {
    pub(crate) fn point_pos(&self, norm_pos: &Vector3<f64>) -> Vector3<f64> {
        self.base + norm_pos * self.size
    }
}

pub struct IsosurfaceSolver<'a> {
    min_depth: usize,
    max_depth: usize,

    cache: EvaluationCache<'a>,
}

impl<'a> IsosurfaceSolver<'a> {
    pub fn new(
        expr: &'a SDFExpression,
        volume: &'a SDFVolume,
        min_depth: usize,
        max_depth: usize,
    ) -> Self {
        Self {
            min_depth,
            max_depth,
            cache: EvaluationCache::new(expr, volume),
        }
    }

    pub fn solve(&mut self) {
        let (volume_cells, face_cells, edge_cells) =
            build_cell_trees(&mut self.cache, self.min_depth, self.max_depth);

        for cell in &volume_cells {
            *cell.cell_data.dual_pos.borrow_mut() =
                DualQuadric::<3>::find_dual(&cell.coord, &cell.subspace, &mut self.cache);
        }
        for cell in &face_cells {
            *cell.cell_data.dual_pos.borrow_mut() =
                DualQuadric::<2>::find_dual(&cell.coord, &cell.subspace, &mut self.cache);
        }
        for cell in &edge_cells {
            *cell.cell_data.dual_pos.borrow_mut() =
                DualQuadric::<1>::find_dual(&cell.coord, &cell.subspace, &mut self.cache);
        }

        let tetras = tetrahedralize(&volume_cells, &face_cells, &edge_cells);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{
        cache::EvaluationCache, cells::build_cell_trees, optimizer::DualQuadric, sdf::SDFExpression,
    };

    use super::SDFVolume;

    #[test]
    fn sphere() {
        let square = (SDFExpression::X * SDFExpression::X
            + SDFExpression::Y * SDFExpression::Y
            + SDFExpression::Z * SDFExpression::Z)
            + (-9.0).into();

        let volume = SDFVolume {
            base: Vector3::new(-5.0, -5.0, -5.0),
            size: 10.0,
        };

        let mut cache = EvaluationCache::new(&square, &volume);

        let (mut volume_cells, _, _) = build_cell_trees(&mut cache, 2, 5);

        for cell in &volume_cells {
            *cell.cell_data.dual_pos.borrow_mut() =
                DualQuadric::<3>::find_dual(&cell.coord, &cell.subspace, &mut cache);
            let len = cell.cell_data.dual_pos.borrow().norm();

            assert!(
                len > 2.99 && len < 3.01,
                "A volume dual was placed too far from the sphere's surface ({}).",
                len - 3.0,
            )
        }
    }
}
