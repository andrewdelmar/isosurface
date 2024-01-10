use crate::{
    cache::EvaluationCache,
    cells::{build_cell_trees, tetrahedralize},
    duals::{find_all_edge_duals, find_all_face_duals, find_all_volume_duals},
    MeshBuffers, SDFVolume, VolumetricFunc,
};

pub struct SolverSettings {
    // The number of additional threads to use for calculations.
    // A value of 0 will not spawn any additional threads.
    pub worker_threads: usize,

    // Octree construction settings.
    pub min_octree_depth: usize,
    pub max_octree_depth: usize,

    // Dual positioning settings.
    pub dual_sample_subdivisions: usize,

    // Tetrahedralization settings.
    pub max_vert_fitting_steps: usize,
    pub vert_fitting_error: f64,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            worker_threads: 0,
            min_octree_depth: 3,
            max_octree_depth: 4,
            dual_sample_subdivisions: 2,
            max_vert_fitting_steps: 32,
            vert_fitting_error: f64::EPSILON,
        }
    }
}

// find_isosurface returns a mesh approximating the isosurface at the 0 value of func.
// The implementation is based on the algorithm described in:
// Isosurfaces Over Simplicial Partitions of Multiresolution Grids by Josiah Manson and Scott Schaefer.
// min and max_depth control the minimum and maximum subdivision of space in each dimension.
pub fn find_isosurface<F>(func: &F, volume: &SDFVolume, settings: &SolverSettings) -> MeshBuffers
where
    F: VolumetricFunc,
{
    let mut cache = EvaluationCache::new(func, volume);

    let (volume_cells, face_cells, edge_cells) = build_cell_trees(
        &mut cache,
        settings.min_octree_depth,
        settings.max_octree_depth,
    );

    find_all_volume_duals(
        &volume_cells,
        &mut cache,
        settings.worker_threads,
        settings.dual_sample_subdivisions,
    );

    find_all_face_duals(
        &face_cells,
        &mut cache,
        settings.worker_threads,
        settings.dual_sample_subdivisions,
    );

    find_all_edge_duals(
        &edge_cells,
        &mut cache,
        settings.worker_threads,
        settings.dual_sample_subdivisions,
    );

    let tetras = tetrahedralize(&volume_cells, &face_cells, &edge_cells);

    let buffers = MeshBuffers::new(
        &mut cache,
        tetras,
        settings.max_vert_fitting_steps,
        settings.vert_fitting_error,
    );

    buffers
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::{find_isosurface, SDFExpression};

    use super::{SDFVolume, SolverSettings};

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

        let mesh = find_isosurface(&sphere, &volume, &SolverSettings::default());

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
