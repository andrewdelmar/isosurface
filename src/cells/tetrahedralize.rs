use crate::{
    partition::PartitionCoord,
    simplex::{Simplex, SimplexVert},
    subspace::{R1Space, R2Space, Subspace},
};

use super::{EdgeCellCollection, FaceCellCollection, VolumeCellCollection};

pub(crate) fn tetrahedralize<'a>(
    volume_cells: &'a VolumeCellCollection,
    face_cells: &'a FaceCellCollection,
    edge_cells: &'a EdgeCellCollection,
) -> Vec<Simplex<'a, 4>> {
    let mut tetras = Vec::new();
    for volume_cell in volume_cells {
        let simplex = Simplex::<1> {
            verts: [SimplexVert::VolumeDual(volume_cell.clone())],
        };
        iterate_faces(
            &volume_cell.coord,
            face_cells,
            edge_cells,
            simplex,
            &mut tetras,
        );
    }
    tetras
}

fn iterate_faces<'a>(
    volume_coord: &PartitionCoord<3>,
    face_cells: &'a FaceCellCollection,
    edge_cells: &'a EdgeCellCollection,
    simplex: Simplex<'a, 1>,
    tetras: &mut Vec<Simplex<'a, 4>>,
) {
    for face_subspace in R2Space::volume_cell_intersections(&volume_coord) {
        let face_coord = face_subspace.project_coord(&volume_coord);

        for face_cell in face_cells.children(&face_coord, &face_subspace) {
            let simplex = simplex.add(SimplexVert::FaceDual(face_cell.clone()));
            iterate_edges(
                &face_cell.coord,
                &face_cell.subspace,
                edge_cells,
                simplex,
                tetras,
            );
        }
    }
}

fn iterate_edges<'a>(
    face_coord: &PartitionCoord<2>,
    face_subspace: &R2Space,
    edge_cells: &'a EdgeCellCollection,
    simplex: Simplex<'a, 2>,
    tetras: &mut Vec<Simplex<'a, 4>>,
) {
    for (edge_coord, edge_subspace) in face_subspace.edges(&face_coord) {
        for edge_cell in edge_cells.children(&edge_coord, &edge_subspace) {
            let simplex = simplex.add(SimplexVert::EdgeDual(edge_cell.clone()));
            iterate_verts(&edge_cell.coord, &edge_cell.subspace, simplex, tetras);
        }
    }
}

fn iterate_verts<'a>(
    coord: &PartitionCoord<1>,
    subspace: &R1Space,
    simplex: Simplex<'a, 3>,
    tetras: &mut Vec<Simplex<'a, 4>>,
) {
    tetras.push(simplex.add(SimplexVert::CellBoundary(
        subspace.unproject_coord(&coord.low_parents()),
    )));

    tetras.push(simplex.add(SimplexVert::CellBoundary(
        subspace.unproject_coord(&coord.high_parents()),
    )));
}
