use crate::{
    cache::EvaluationCache,
    cells::{build_cell_trees, tetrahedralize},
    optimizer::DualQuadric,
    sdf::SDFExpression,
    volume::SDFVolume,
    MeshBuffers,
};

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

    pub fn build_mesh(&mut self) -> MeshBuffers {
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
        MeshBuffers::new(&mut self.cache, tetras)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{sdf::SDFExpression, IsosurfaceSolver};

    use super::SDFVolume;

    #[test]
    fn sphere() {
        let sphere = (SDFExpression::X * SDFExpression::X
            + SDFExpression::Y * SDFExpression::Y
            + SDFExpression::Z * SDFExpression::Z)
            + (-9.0).into();

        let volume = SDFVolume {
            base: Vector3::new(-5.0, -5.0, -5.0),
            size: Vector3::new(10.0, 10.0, 10.0),
        };

        let mut solver = IsosurfaceSolver::new(&sphere, &volume, 2, 5);

        let mesh = solver.build_mesh();

        for vert in mesh.0 {
            let len = vert.norm();
            assert!(
                len > 2.99 && len < 3.01,
                "A mesh vertex was placed too far from the sphere's surface ({}).",
                len - 3.0,
            )
        }
    }
}
