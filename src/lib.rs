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
mod volume;

pub use data::{sdf::SDFExpression, Dimension, VolumetricFunc};
pub use isosurface::find_isosurface;
pub use mesh::MeshBuffers;
