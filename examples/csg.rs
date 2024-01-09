#![feature(generic_const_exprs)]

use std::fs::File;

use isosurface_simplex::{find_isosurface, SDFExpression, SDFVolume};
use nalgebra::Vector3;

fn sphere(pos: Vector3<f64>, size: f64) -> SDFExpression {
    let x = SDFExpression::x() - pos.x.into();
    let y = SDFExpression::y() - pos.y.into();
    let z = SDFExpression::z() - pos.z.into();

    (x.clone() * x + y.clone() * y + z.clone() * z) + (-size * size).into()
}

fn main() {
    let a = sphere(Vector3::new(1.0, 1.0, 2.0), 2.0);
    let b = sphere(Vector3::new(3.0, 1.0, 2.0), 1.0);

    let buffers = find_isosurface(
        &SDFExpression::min(a, b),
        &SDFVolume {
            base: Vector3::new(-5.0, -5.0, -5.0),
            size: Vector3::new(10.0, 10.0, 10.0),
        },
        4,
        5,
    );

    let mut file = File::create("csg.obj").expect("Failed to open output file");
    buffers
        .export_obj(&mut file)
        .expect("Failed to export model");
}
