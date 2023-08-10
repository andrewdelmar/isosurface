use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use crate::MeshBuffers;

pub fn export_obj(path: &Path, mesh: &MeshBuffers) -> Result<(), io::Error> {
    let MeshBuffers(verts, inds) = mesh;
    let file = File::create(path)?;

    let mut buff = BufWriter::new(file);

    for vert in verts {
        let line = format!("v {} {} {}\n", vert.x, vert.y, vert.z);
        buff.write(line.as_bytes())?;
    }

    for chunk in inds.chunks_exact(3) {
        let line = format!("f {} {} {}\n", chunk[0], chunk[1], chunk[2]);
        buff.write(line.as_bytes())?;
    }

    buff.flush()?;

    Ok(())
}
