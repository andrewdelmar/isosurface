#![feature(generic_const_exprs, adt_const_params, box_patterns, let_chains)]

mod cache;
mod cells;
mod data;
mod isosurface;
mod mesh;
mod optimizer;
mod partition;
mod simplex;
mod subspace;

pub use data::{sdf::SDFExpression, Dimension, SDFVolume, VolumetricFunc};
pub use isosurface::{find_isosurface, SolverSettings};
pub use mesh::MeshBuffers;
