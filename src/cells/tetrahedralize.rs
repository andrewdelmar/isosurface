use crate::{
    partition::PartitionCoord,
    simplex::{Simplex, Vertex},
    subspace::{R1Space, R2Space, Subspace},
};

use super::{EdgeCellCollection, FaceCellCollection, VolumeCellCollection};

//TODO this could return an iterator instead.
pub(crate) fn tetrahedralize<'a>(
    volume_cells: &'a VolumeCellCollection,
    face_cells: &'a FaceCellCollection,
    edge_cells: &'a EdgeCellCollection,
) -> Vec<Simplex<'a, 4>> {
    let mut tetras = Vec::new();
    for volume_cell in volume_cells {
        for tri in tris_in_volume_cell(&volume_cell.coord, face_cells, edge_cells) {
            tetras.push(tri.add(Vertex::VolumeDual(volume_cell.clone())))
        }
    }
    tetras
}

fn tris_in_volume_cell<'a>(
    coord: &PartitionCoord<3>,
    face_cells: &'a FaceCellCollection,
    edge_cells: &'a EdgeCellCollection,
) -> Vec<Simplex<'a, 3>> {
    let mut tris = Vec::new();
    for face_subspace in R2Space::volume_cell_intersections(&coord) {
        let volume_face_coord = face_subspace.project_coord(&coord);
        for face_cell in face_cells.children(&volume_face_coord, &face_subspace) {
            for edge in edges_in_face_cell(&face_cell.coord, &face_cell.subspace, edge_cells) {
                tris.push(edge.add(Vertex::FaceDual(face_cell.clone())))
            }
        }
    }

    tris
}

fn edges_in_face_cell<'a>(
    coord: &PartitionCoord<2>,
    subspace: &R2Space,
    edge_cells: &'a EdgeCellCollection,
) -> Vec<Simplex<'a, 2>> {
    let mut edges = Vec::new();
    for (face_edge_coord, edge_subspace) in subspace.edges(&coord) {
        for edge_cell in edge_cells.children(&face_edge_coord, &edge_subspace) {
            for vert in verts_in_edge_cell(&edge_cell.coord, &edge_cell.subspace) {
                edges.push(vert.add(Vertex::EdgeDual(edge_cell.clone())));
            }
        }
    }

    edges
}

fn verts_in_edge_cell<'a>(coord: &PartitionCoord<1>, subspace: &R1Space) -> Vec<Simplex<'a, 1>> {
    let h = Simplex::<'a, 1> {
        verts: [Vertex::CellBoundary(
            subspace.unproject_coord(&coord.high_parents()),
        )],
    };
    let l = Simplex::<'a, 1> {
        verts: [Vertex::CellBoundary(
            subspace.unproject_coord(&coord.low_parents()),
        )],
    };
    return vec![l, h];
}