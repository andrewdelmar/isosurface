use std::{
    collections::HashMap,
    io::{self, Write},
};

use nalgebra::Vector3;

use crate::{
    cache::EvaluationCache,
    simplex::{Simplex, SimplexVert},
};

// A MeshBuffers struct contains an index and vertex buffer representing an isosurface.
pub struct MeshBuffers(pub Vec<Vector3<f64>>, pub Vec<usize>);

impl MeshBuffers {
    pub(crate) fn new<'a>(cache: &mut EvaluationCache, tetras: Vec<Simplex<'a, 4>>) -> Self {
        let tetras = Self::marching_tetrahedra(cache, tetras);
        Self::collect_buffers(&tetras, cache)
    }

    fn marching_tetrahedra<'a>(
        cache: &mut EvaluationCache,
        tetras: Vec<Simplex<'a, 4>>,
    ) -> Vec<Face<'a>> {
        let mut faces = Vec::<Face>::with_capacity(tetras.len() * 2);

        for tetra in tetras {
            faces.append(&mut Self::tetra_tris(cache, &tetra))
        }

        faces
    }

    // TODO using a vec here leads to unnecessary heap allocations.
    // This probably isnt a performance issue but an arrayvec or enum would be cleaner.
    // Using bumpalo or some arena allocator should also help.
    fn tetra_tris<'a>(cache: &mut EvaluationCache, tetra: &Simplex<'a, 4>) -> Vec<Face<'a>> {
        let [a, b, c, d] = &tetra.verts;
        let [ai, bi, ci, di] = [
            a.inside(cache),
            b.inside(cache),
            c.inside(cache),
            d.inside(cache),
        ];

        match [ai, bi, ci, di] {
            [true, true, true, true] | [false, false, false, false] => Vec::new(),

            [true, true, true, false] => Face::tri((a, d), (b, d), (c, d)),
            [false, false, false, true] => Face::tri((d, a), (d, b), (d, c)),

            [true, true, false, true] => Face::tri((a, c), (b, c), (d, c)),
            [false, false, true, false] => Face::tri((c, a), (c, b), (c, d)),

            [true, false, true, true] => Face::tri((a, b), (c, b), (d, b)),
            [false, true, false, false] => Face::tri((b, a), (b, c), (b, d)),

            [false, true, true, true] => Face::tri((b, a), (c, a), (d, a)),
            [true, false, false, false] => Face::tri((a, b), (a, c), (a, d)),

            [true, true, false, false] => Face::quad((a, c), (a, d), (b, d), (b, c)),
            [false, false, true, true] => Face::quad((c, a), (d, a), (d, b), (c, b)),

            [true, false, true, false] => Face::quad((a, b), (a, d), (c, d), (c, b)),
            [false, true, false, true] => Face::quad((b, a), (d, a), (d, c), (b, c)),

            [true, false, false, true] => Face::quad((a, b), (a, c), (d, c), (d, b)),
            [false, true, true, false] => Face::quad((b, a), (c, a), (c, d), (b, d)),
        }
    }

    fn collect_buffers<'a>(faces: &Vec<Face<'a>>, cache: &mut EvaluationCache) -> Self {
        let mut verts = Vec::<Vector3<f64>>::new();
        let mut inds = Vec::<usize>::new();
        let mut ind_cache = HashMap::<FaceVert<'a>, usize>::new();

        for face in faces {
            for vert in &face.0 {
                let ind = ind_cache.entry(vert.clone()).or_insert_with(|| {
                    verts.push(vert.crossing(cache));
                    verts.len() - 1
                });

                inds.push(*ind);
            }
        }

        Self(verts, inds)
    }

    pub fn export_obj<W: Write>(&self, writer: &mut W) -> Result<(), io::Error> {
        let Self(verts, inds) = self;

        for vert in verts {
            let line = format!("v {} {} {}\n", vert.x, vert.y, vert.z);
            writer.write(line.as_bytes())?;
        }

        for chunk in inds.chunks_exact(3) {
            let line = format!("f {} {} {}\n", chunk[0], chunk[1], chunk[2]);
            writer.write(line.as_bytes())?;
        }

        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FaceVert<'a> {
    i: SimplexVert<'a>,
    o: SimplexVert<'a>,
}

impl<'a> FaceVert<'a> {
    fn crossing(&self, cache: &mut EvaluationCache) -> Vector3<f64> {
        // TODO This should use a couple of iterations of newtons method or something.
        // Maybe pass in some error value/max iterations to optimize to.
        // Placing vertices on the isosurface will always under/overshoot concave/convex curves.
        // This would ideally consider some estimate of error across neighboring tris.
        let iv = self.i.eval(cache);
        let ov = self.o.eval(cache);
        let t = (-iv / (ov - iv)).clamp(0.0, 1.0);
        self.i.pos(cache) * (1.0 - t) + self.o.pos(cache) * t
    }
}

impl<'a> From<(&SimplexVert<'a>, &SimplexVert<'a>)> for FaceVert<'a> {
    fn from(value: (&SimplexVert<'a>, &SimplexVert<'a>)) -> Self {
        let (i, o) = value;
        FaceVert {
            i: i.clone(),
            o: o.clone(),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Face<'a>([FaceVert<'a>; 3]);

impl<'a> Face<'a> {
    fn tri<I>(a: I, b: I, c: I) -> Vec<Face<'a>>
    where
        I: Into<FaceVert<'a>>,
    {
        vec![Face([a.into(), b.into(), c.into()])]
    }

    fn quad<I>(a: I, b: I, c: I, d: I) -> Vec<Face<'a>>
    where
        I: Into<FaceVert<'a>>,
    {
        let (ai, bi, ci, di) = (a.into(), b.into(), c.into(), d.into());

        // TODO this should avoid skinny tris.
        vec![
            Face([ai.clone(), bi, ci.clone()]),
            Face([ci.clone(), di, ai.clone()]),
        ]
    }
}
