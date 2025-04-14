use render_engine as re;

use re::object::ObjectPrototype;
use re::render_passes;
use re::system::{Pass, System};
use re::window::Window;
use re::mesh::{PrimitiveTopology, Mesh};

use std::collections::HashMap;

use tests_render_engine::mesh::{VPosColor2D, fullscreen_quad};
use tests_render_engine::relative_path;

fn main() {
    // initialize window
    let (mut window, queue) = Window::new();
    let device = queue.device().clone();

    // create system
    let render_pass = render_passes::basic(device.clone());
    let mut system = System::new(
        queue.clone(),
        vec![
            Pass {
                name: "geometry",
                images_created_tags: vec!["geo"],
                images_needed_tags: vec![],
                render_pass: render_pass.clone(),
            },
            Pass {
                name: "postprocess",
                images_created_tags: vec!["final"],
                images_needed_tags: vec!["geo"],
                render_pass: render_pass.clone(),
            },
        ],
        // custom images, we use none
        HashMap::new(),
        "final",
    );

    window.set_render_pass(render_pass.clone());

    // there are 2 objects: the triangle and a fullscreen quad used to do
    // postprocessing in a separate pass
    let triangle = ObjectPrototype {
        vs_path: relative_path("shaders/multipass/tri_vert.glsl"),
        fs_path: relative_path("shaders/multipass/tri_frag.glsl"),
        fill_type: PrimitiveTopology::TriangleList,
        read_depth: false,
        write_depth: false,
        mesh: Mesh {
            vertices: vec![
                VPosColor2D {
                    position: [0.0, -1.0],
                    color: [1.0, 0.0, 0.0],
                },
                VPosColor2D {
                    position: [-1.0, 1.0],
                    color: [0.0, 1.0, 0.0],
                },
                VPosColor2D {
                    position: [1.0, 1.0],
                    color: [0.0, 0.0, 1.0],
                },
            ],
            indices: vec![0, 1, 2],
        },
        collection: (),
        custom_dynamic_state: None,
    }
    .build(queue.clone(), render_pass.clone(), 1);

    let quad = fullscreen_quad(
        queue.clone(),
        render_pass.clone(),
        relative_path("shaders/multipass/postpro_vert.glsl"),
        relative_path("shaders/multipass/postpro_frag.glsl"),
    );

    while !window.update() {
        // draw
        system.start_window(&mut window);
        system.add_object(&triangle);
        system.next_pass();
        system.add_object(&quad);
        system.finish_to_window(&mut window);
    }

    println!("FPS: {}", window.get_fps());
    system.print_stats();
}
