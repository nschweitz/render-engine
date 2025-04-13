use render_engine as re;

use re::collection::{Set, Data, CollectionData};
use re::collection_cache::pds_for_buffers;
use re::mesh::{PrimitiveTopology, Vertex};
use re::object::{ObjectPrototype, Object, Drawcall};
use re::pipeline_cache::PipelineSpec;
use re::system::{Pass, System};
use re::window::Window;
use re::{render_passes, Format, Image, Pipeline, Queue, RenderPass};

use vulkano::command_buffer::DynamicState;
use vulkano::pipeline::viewport::Viewport;

use nalgebra_glm::*;

use std::collections::HashMap;
use std::sync::Arc;

use tests_render_engine::mesh::{convert_meshes, fullscreen_quad, load_obj};
use tests_render_engine::{relative_path, OrbitCamera, Matrix4};

// patches are laid out in a 6x1
const SHADOW_MAP_DIMS: [u32; 2] = [6144, 1024];
const PATCH_DIMS: [f32; 2] = [1024.0, 1024.0];

fn main() {
    // initialize window
    let (mut window, queue) = Window::new();
    let device = queue.device().clone();

    // create system
    let patched_shadow_image: Image = vulkano::image::AttachmentImage::sampled(
        device.clone(),
        SHADOW_MAP_DIMS,
        Format::D32Sfloat,
    )
    .unwrap();
    let mut custom_images = HashMap::new();
    custom_images.insert("shadow_map", patched_shadow_image);

    let rpass1 = render_passes::only_depth(device.clone());
    let rpass2 = render_passes::basic(device.clone());
    let rpass3 = render_passes::with_depth(device.clone());

    let mut system = System::new(
        queue.clone(),
        vec![
            // renders to shadow cubemap
            Pass {
                name: "shadow",
                images_created_tags: vec!["shadow_map"],
                images_needed_tags: vec![],
                render_pass: rpass1.clone(),
            },
            // displays shadow map for debugging
            Pass {
                name: "cubemap_view",
                images_created_tags: vec!["cubemap_view"],
                images_needed_tags: vec!["shadow_map"],
                render_pass: rpass2.clone(),
            },
            // renders final scene
            Pass {
                name: "final",
                images_created_tags: vec!["final_color", "final_depth"],
                images_needed_tags: vec!["shadow_map"],
                render_pass: rpass3.clone(),
            },
        ],
        custom_images,
        "final_color",
    );
    window.set_render_pass(rpass1.clone());

    // create buffer and set for model matrix
    let model_data: Matrix4 = Mat4::identity().into();

    // initialize camera
    let mut camera = OrbitCamera::default();

    // load object
    let (mut models, _materials) =
        load_obj(&relative_path("meshes/shadowtest.obj")).expect("Couldn't load OBJ file");
    let mesh = convert_meshes(&[models.remove(0)]).remove(0);

    let mut final_object = ObjectPrototype {
        vs_path: relative_path("shaders/point-shadow/shadow_cast_vert.glsl"),
        fs_path: relative_path("shaders/point-shadow/shadow_cast_frag.glsl"),
        fill_type: PrimitiveTopology::TriangleList,
        read_depth: true,
        write_depth: true,
        mesh,
        collection: (
        ),
        custom_dynamic_state: None,
    }
    .build(queue.clone(), rpass3.clone());

    // create fullscreen quad to debug cubemap
    let quad = fullscreen_quad(
        queue.clone(),
        rpass2.clone(),
        relative_path("shaders/point-shadow/display_cubemap_vert.glsl"),
        relative_path("shaders/point-shadow/display_cubemap_frag.glsl"),
    );

    // load dragon
    let (mut models, _materials) =
        load_obj(&relative_path("meshes/raptor.obj")).expect("Couldn't load OBJ file");
    let mesh = convert_meshes(&[models.remove(0)]).remove(0);

    let mut base_object = ObjectPrototype {
        vs_path: relative_path("shaders/point-shadow/shadow_cast_vert.glsl"),
        fs_path: relative_path("shaders/point-shadow/shadow_cast_frag.glsl"),
        fill_type: PrimitiveTopology::TriangleList,
        read_depth: true,
        write_depth: true,
        mesh,
        collection: (),
        custom_dynamic_state: None,
    };

    // create 6 different dragon objects, each with a different view matrix and
    // dynamic state, to draw to the 6 different faces of the patched texture
    let shadow_casters = convert_to_shadow_casters(queue.clone(), rpass1.clone(),
        base_object.clone());

    // create a version of the base object with shaders for rendering the
    // final image
    let object_final = ObjectPrototype {
        vs_path: relative_path("shaders/point-shadow/final_vert.glsl"),
        fs_path: relative_path("shaders/point-shadow/final_frag.glsl"),
        // FIXME: Collections has to somehow end up with depth sampler here
        ..base_object
    }
    .build(queue.clone(), rpass3.clone());

    let pipeline_final = object_final.pipeline_spec.concrete(device.clone(), rpass3);

    // used in main loop
    // If we don't make this dyn, it breaks because shadow_casters and quad have different type
    // thingies: shadow_casters is Object<..., ..., ..., ...>, quad is Object<()>
    let mut all_objects: HashMap<&str, Vec<Arc<dyn Drawcall>>> = HashMap::new();
    all_objects.insert("shadow", shadow_casters);
    all_objects.insert("cubemap_view", vec![quad]);

    while !window.update() {
        // update camera and camera buffer
        camera.update(window.get_frame_info());
        let camera_buffer = camera.get_buffer(queue.clone());
        let camera_set = pds_for_buffers(pipeline_final.clone(), &[camera_buffer], 1).unwrap();

        if window.get_frame_info().keys_down.c {
            system.output_tag = "cubemap_view";
        } else {
            system.output_tag = "final_color";
        }

        // create updated object of final pass
        // it already has a model buffer in custom_sets, just need to add the
        // camera set
        let mut cur_object_final = object_final.clone();
        cur_object_final.custom_sets.push(camera_set);

        // add to scene
        all_objects.insert("final", vec![cur_object_final]);

        // draw
        system.render_to_window(&mut window, all_objects.clone());
    }

    println!("FPS: {}", window.get_fps());
}

fn convert_to_shadow_casters<V: Vertex, D: CollectionData>(
    queue: Queue,
    rpass: RenderPass,
    base_object: ObjectPrototype<V, D>,
) -> Vec<Object<(Set<(Matrix4,)>, Set<(Matrix4,)>, Set<(Matrix4,)>, Set<(Light,)>)>> {
    // if you want to make point lamps cast shadows, you need shadow cubemaps
    // render-engine doesn't support geometry shaders, so the easiest way to do
    // this is to convert one object into 6 different ones, one for each face of
    // the cubemap, that each render to a different part of a 2D texture.
    // for now this function assumes a 6x1 patch layout
    let view_directions = [
        vec3(1.0, 0.0, 0.0),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
    ];

    let up_directions = [
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
    ];

    let patch_positions = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [5.0, 0.0],
    ];

    let (near, far) = (1.0, 250.0);
    // pi / 2 = 90 deg., 1.0 = aspect ratio
    // we use a fov 1% too big to make sure sampling doesn't go between patches
    let proj_data: Matrix4 = perspective(1.0, std::f32::consts::PI / 2.0 * 1.01, near, far).into();

    let model_data: Matrix4 = scale(&Mat4::identity(), &vec3(0.1, 0.1, 0.1)).into();

    view_directions
        .iter()
        .zip(&up_directions)
        .zip(&patch_positions)
        .map(|((dir, up), patch_pos): ((&Vec3, &Vec3), &[f32; 2])| {
            let light_pos = vec3(0, 0, 0);
            let view_data: Matrix4 = look_at(&light_pos, &(light_pos + dir), up).into();

            // dynamic state for the current cubemap face, represents which part
            // of the patched texture we draw to
            let margin = 0.0;
            let origin = [
                patch_pos[0] * PATCH_DIMS[0] + margin,
                patch_pos[1] * PATCH_DIMS[1] + margin,
            ];
            let dynamic_state = dynamic_state_for_bounds(
                origin,
                [PATCH_DIMS[0] - margin * 2.0, PATCH_DIMS[1] - margin * 2.0],
            );

            ObjectPrototype {
                collection: (
                    (model_data,),
                    (proj_data,),
                    (view_data,),
                    (light_pos,),
                ),
                custom_dynamic_state: Some(dynamic_state),
                ..base_object
            }
            .build(queue.clone(), rpass.clone())
        })
        .collect()
}

fn create_projection_set(queue: Queue, pipeline: Pipeline) -> re::Set {
    let (near, far) = (1.0, 250.0);
    // pi / 2 = 90 deg., 1.0 = aspect ratio
    let proj_data: [[f32; 4]; 4] = perspective(1.0, std::f32::consts::PI / 2.0, near, far).into();
    let proj_buffer = bufferize_data(queue, proj_data);

    pds_for_buffers(pipeline, &[proj_buffer], 1).unwrap()
}

fn dynamic_state_for_bounds(origin: [f32; 2], dimensions: [f32; 2]) -> DynamicState {
    DynamicState {
        line_width: None,
        viewports: Some(vec![Viewport {
            origin,
            dimensions,
            depth_range: 0.0..1.0,
        }]),
        scissors: None,
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
struct Light {
    position: [f32; 4],
    strength: [f32; 4],
}
impl Data for Light {}

#[derive(Default, Debug, Clone, Copy)]
struct V2D {
    position: [f32; 2],
}
vulkano::impl_vertex!(V2D, position);
