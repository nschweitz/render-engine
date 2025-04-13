/*
Terminology:
Mesh: vertices and indices, nothing else
Object: mesh + other stuff.
 */

use render_engine::mesh::{Mesh, PrimitiveTopology, Vertex};
use render_engine::utils::load_texture;
use render_engine::{Format, Queue, Image, RenderPass};
use render_engine::object::{ObjectPrototype, Object};
use render_engine::impl_vertex;

use crate::relative_path;

use nalgebra_glm::*;

use std::path::{Path, PathBuf};

pub use tobj::load_obj;

pub fn convert_meshes(models: &[tobj::Model]) -> Vec<Mesh<VPosTexNorm>> {
    // converts all provided into meshes of type VPosTexNorm, which includes all
    // information commonly incldued in obj files: positions, texture
    // coordinates and normals.
    models
        .iter()
        .map(|model| convert_mesh(&model.mesh))
        .collect()
}

pub fn load_textures(
    queue: Queue,
    root_path: &Path,
    materials: &[tobj::Material],
) -> Vec<(Image, Image, Image)> {
    // loads all textures for all materials provided by returning 3 images for
    // each material: a diffuse texture, a specular texture, and a normal
    // texture, in that order
    materials
        .iter()
        .map(|mat| {
            // diffuse
            let diff_path = {
                let maybe_path = root_path.join(Path::new(&mat.diffuse_texture));

                // if the diffuse texture path is empty or the file doesn't
                // exist, use a placeholder
                if mat.diffuse_texture == "" {
                    println!("{} has no diffuse texture", mat.name);
                    relative_path("textures/missing.png")
                } else if !maybe_path.exists() {
                    println!(
                        "{} diffuse texture does not exist: {:?}",
                        mat.name, maybe_path
                    );
                    relative_path("textures/missing.png")
                } else {
                    maybe_path
                }
            };

            // specular
            let spec_path = {
                let maybe_path = root_path.join(Path::new(&mat.specular_texture));

                if mat.specular_texture == "" {
                    println!("{} has no specular texture", mat.name);
                    relative_path("textures/missing-spec.png")
                } else if !maybe_path.exists() {
                    println!(
                        "{} specular texture does not exist: {:?}",
                        mat.name, maybe_path
                    );
                    relative_path("textures/missing-spec.png")
                } else {
                    maybe_path
                }
            };

            // normal
            let normal_path = {
                let maybe_path = root_path.join(Path::new(&mat.normal_texture));

                if mat.normal_texture == "" {
                    println!("{} has no normal texture", mat.name);
                    relative_path("textures/missing-normal.png")
                } else if !maybe_path.exists() {
                    println!(
                        "{} normal texture does not exist: {:?}",
                        mat.name, maybe_path
                    );
                    relative_path("textures/missing-normal.png")
                } else {
                    maybe_path
                }
            };

            let diff_tex = load_texture(queue.clone(), &diff_path, Format::R8G8B8A8Srgb);
            let spec_tex = load_texture(queue.clone(), &spec_path, Format::R8G8B8A8Unorm);
            let norm_tex = load_texture(queue.clone(), &normal_path, Format::R8G8B8A8Unorm);

            (diff_tex, spec_tex, norm_tex)
        })
        .collect()
}

pub fn convert_mesh(mesh: &tobj::Mesh) -> Mesh<VPosTexNorm> {
    // converts a tobj mesh to one of vertices render-engine will be able to use
    let mut vertices: Vec<VPosTexNorm> = vec![];

    for i in 0..mesh.positions.len() / 3 {
        let pos = [
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        ];
        let normal = [
            mesh.normals[i * 3],
            mesh.normals[i * 3 + 1],
            mesh.normals[i * 3 + 2],
        ];
        // if no texture coordinates are found, use a dummy value
        // TODO: let the user specify how lenient they want to be with this
        let tex_coord = if mesh.texcoords.len() <= i * 2 + 1 {
            [0.0, 0.0]
        } else {
            [mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1] * -1.0]
        };

        let vertex = VPosTexNorm {
            position: pos,
            tex_coord,
            normal,
        };

        vertices.push(vertex);
    }

    Mesh {
        vertices,
        indices: mesh.indices.clone(),
    }
}

pub fn add_tangents_multi(meshes: &[Mesh<VPosTexNorm>]) -> Vec<Mesh<VPosTexNormTan>> {
    meshes.iter().map(|mesh| add_tangents(mesh)).collect()
}

pub fn add_tangents(mesh: &Mesh<VPosTexNorm>) -> Mesh<VPosTexNormTan> {
    // use to compute tangents for a mesh with normals and texture coordinates
    let (vertices, indices) = (&mesh.vertices, &mesh.indices);

    let mut tangents: Vec<Vec3> = vec![vec3(0.0, 0.0, 0.0); vertices.len()];

    for i in 0..indices.len() / 3 {
        let face = [
            vertices[indices[i * 3] as usize],
            vertices[indices[i * 3 + 1] as usize],
            vertices[indices[i * 3 + 2] as usize],
        ];
        let (tangent, _bitangent) = tangent_bitangent_for_face(&face);
        tangents[indices[i * 3] as usize] += tangent;
        tangents[indices[i * 3 + 1] as usize] += tangent;
        tangents[indices[i * 3 + 2] as usize] += tangent;
    }

    let new_vertices: Vec<VPosTexNormTan> = vertices
        .iter()
        .enumerate()
        .map(|(idx, v)| {
            let t = normalize(&tangents[idx]);

            VPosTexNormTan {
                position: v.position,
                tex_coord: v.tex_coord,
                normal: v.normal,
                tangent: t.into(),
            }
        })
        .collect();

    Mesh {
        vertices: new_vertices,
        indices: indices.clone(),
    }
}

pub fn fullscreen_quad(queue: Queue, render_pass: RenderPass, vs_path: PathBuf, fs_path: PathBuf) -> Object<()> {
    ObjectPrototype {
        vs_path,
        fs_path,
        fill_type: PrimitiveTopology::TriangleStrip,
        read_depth: false,
        write_depth: false,
        mesh: Mesh {
            vertices: vec![
                VPos2D {
                    position: [-1.0, -1.0],
                },
                VPos2D {
                    position: [-1.0, 1.0],
                },
                VPos2D {
                    position: [1.0, -1.0],
                },
                VPos2D {
                    position: [1.0, 1.0],
                },
            ],
            indices: vec![0, 1, 2, 3],
        },
        collection: (),
        custom_dynamic_state: None,
    }
    .build(queue, render_pass, 0)
}

pub fn wireframe(mesh: &Mesh<VPos>) -> Mesh<VPos> {
    // converts a mesh of triangles into one with lines for every edge, suitable
    // for drawing a wireframe version of a mesh

    let mut vertices = vec![];
    let mut indices = vec![];

    for i in 0..mesh.indices.len() / 3 {
        let v1 = mesh.vertices[mesh.indices[3 * i] as usize];
        let v2 = mesh.vertices[mesh.indices[3 * i + 1] as usize];
        let v3 = mesh.vertices[mesh.indices[3 * i + 2] as usize];
        vertices.push(v1);
        vertices.push(v2);
        vertices.push(v3);
        // line 1, between v1 and v2
        indices.push(3 * i as u32);
        indices.push(3 * i as u32 + 1);
        // line 2, between v2 and v3
        indices.push(3 * i as u32 + 1);
        indices.push(3 * i as u32 + 2);
        // line 3, between v3 and v1
        indices.push(3 * i as u32 + 2);
        indices.push(3 * i as u32);
    }

    Mesh { vertices, indices }
}

pub fn merge<V: Vertex + Clone>(meshes: &[Mesh<V>]) -> Mesh<V> {
    // merges a list of meshes into a single mesh

    // you could probably write this as an iterator, i'm just too lazy
    let mut vertices = vec![];
    let mut indices = vec![];
    // we need to offset some indices because the vertices are being merged into
    // one giant list
    let mut index_offset = 0;
    for mesh in meshes.iter() {
        for vertex in mesh.vertices.iter().cloned() {
            vertices.push(vertex);
        }

        for index in mesh.indices.iter() {
            indices.push(index + index_offset);
        }

        index_offset += mesh.vertices.len() as u32;
    }

    Mesh { vertices, indices }
}

fn tangent_bitangent_for_face(face: &[VPosTexNorm; 3]) -> (Vec3, Vec3) {
    let (v1, v2, v3) = (
        make_vec3(&face[0].position),
        make_vec3(&face[1].position),
        make_vec3(&face[2].position),
    );
    let (n1, n2, n3) = (
        make_vec3(&face[0].normal),
        make_vec3(&face[1].normal),
        make_vec3(&face[2].normal),
    );
    let (uv1, uv2, uv3) = (
        make_vec2(&face[0].tex_coord),
        make_vec2(&face[1].tex_coord),
        make_vec2(&face[2].tex_coord),
    );

    // compute average normal of vertices
    let normal = normalize(&(n1 + n2 + n3));

    // calculate edge length and UV differences
    let edge1 = v2 - v1;
    let edge2 = v3 - v1;
    let duv1 = uv2 - uv1;
    let duv2 = uv3 - uv1;

    // compute and bitangent
    let mut tangent = normalize(&vec3(
        duv2.y * edge1.x - duv1.y * edge2.x,
        duv2.y * edge1.y - duv1.y * edge2.y,
        duv2.y * edge1.z - duv1.y * edge2.z,
    ));

    tangent = normalize(&(tangent - dot(&tangent, &normal) * normal));
    let bitangent = tangent.cross(&normal);

    (tangent, bitangent)
}

// using From and Into gets kinda messy cause mesh is another crate :(
pub fn only_pos_from_ptnt(mesh: &Mesh<VPosTexNormTan>) -> Mesh<VPos> {
    let vertices: Vec<VPos> = mesh
        .vertices
        .iter()
        .map(|vertex| VPos {
            position: vertex.position,
        })
        .collect();

    Mesh {
        vertices,
        indices: mesh.indices.clone(),
    }
}

pub fn only_pos(mesh: &Mesh<VPosTexNorm>) -> Mesh<VPos> {
    let vertices: Vec<VPos> = mesh
        .vertices
        .iter()
        .map(|vertex| VPos {
            position: vertex.position,
        })
        .collect();

    Mesh {
        vertices,
        indices: mesh.indices.clone(),
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct VPos {
    pub position: [f32; 3],
}
impl_vertex!(VPos, position);

#[derive(Default, Debug, Clone, Copy)]
pub struct VPos2D {
    pub position: [f32; 2],
}
impl_vertex!(VPos2D, position);

#[derive(Default, Debug, Clone, Copy)]
pub struct VPosColor2D {
    pub position: [f32; 2],
    pub color: [f32; 3],
}
impl_vertex!(VPosColor2D, position, color);

#[derive(Default, Debug, Clone, Copy)]
pub struct VPosTexNorm {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normal: [f32; 3],
}
impl_vertex!(VPosTexNorm, position, tex_coord, normal);

#[derive(Default, Debug, Clone, Copy)]
pub struct VPosTexNormTan {
    pub position: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
}
impl_vertex!(VPosTexNormTan, position, tex_coord, normal, tangent);
