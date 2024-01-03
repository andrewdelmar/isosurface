use crate::{
    cache::EvaluationCache,
    cells::{build_cell_trees, tetrahedralize},
    optimizer::{find_edge_dual, find_face_dual, find_volume_dual},
    volume::SDFVolume,
    MeshBuffers, VolumetricFunc,
};

// find_isosurface returns a mesh approximating the isosurface at the 0 value of expr.
// The implementation is based on the algorithm described in:
// Isosurfaces Over Simplicial Partitions of Multiresolution Grids by Josiah Manson and Scott Schaefer.
// min and max_depth control the minimum and maximum subdivision of space in each dimension.
pub fn find_isosurface<F>(
    func: &F,
    volume: &SDFVolume,
    min_depth: usize,
    max_depth: usize,
) -> MeshBuffers
where
    F: VolumetricFunc,
{
    let mut cache = EvaluationCache::new(func, volume);

    let (volume_cells, face_cells, edge_cells) = build_cell_trees(&mut cache, min_depth, max_depth);

    for cell in &volume_cells {
        *cell.cell_data.dual_pos.borrow_mut() =
            find_volume_dual(&cell.coord, &cell.subspace, &mut cache);
    }
    for cell in &face_cells {
        *cell.cell_data.dual_pos.borrow_mut() =
            find_face_dual(&cell.coord, &cell.subspace, &mut cache);
    }
    for cell in &edge_cells {
        *cell.cell_data.dual_pos.borrow_mut() =
            find_edge_dual(&cell.coord, &cell.subspace, &mut cache);
    }

    let tetras = tetrahedralize(&volume_cells, &face_cells, &edge_cells);

    MeshBuffers::new(&mut cache, tetras)
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{find_isosurface, SDFExpression};

    use super::SDFVolume;

    #[test]
    fn sphere() {
        let sphere = (SDFExpression::x() * SDFExpression::x()
            + SDFExpression::y() * SDFExpression::y()
            + SDFExpression::z() * SDFExpression::z())
            + (-9.0).into();

        let volume = SDFVolume {
            base: Vector3::new(-5.0, -5.0, -5.0),
            size: Vector3::new(10.0, 10.0, 10.0),
        };

        let mesh = find_isosurface(&sphere, &volume, 2, 3);

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
