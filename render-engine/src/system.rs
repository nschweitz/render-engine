use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::framebuffer::{
    AttachmentDescription, Framebuffer, FramebufferAbstract, RenderPassAbstract,
};
use vulkano::image::{AttachmentImage, ImageViewAccess};
use vulkano::pipeline::viewport::Viewport;
use vulkano::sync::GpuFuture;

use std::collections::HashMap;
use std::sync::Arc;

use crate::collection_cache::CollectionCache;
use crate::object::Drawcall;
use crate::pipeline_cache::PipelineCache;
use crate::render_passes::clear_values_for_pass;
use crate::utils::Timer;
use crate::window::Window;

// TODO: make the whole thing less prone to runtime panics. vecs of strings are
// a little sketchy. Maybe make a function that checks the system to ensure
// it'll work?

// A system is a list of passes that takes a bunch of data and produces a frame
// for it.
pub struct System<'a> {
    pub passes: Vec<Pass<'a>>,
    pipeline_caches: Vec<PipelineCache>,
    collection_cache: CollectionCache,
    // stores the vbuf of the screen-filling square used for non-geometry passes
    device: Arc<Device>,
    queue: Arc<Queue>,
    pub output_tag: &'a str,
    cached_images: Option<HashMap<String, Arc<dyn ImageViewAccess + Send + Sync>>>,
    pub custom_images: HashMap<&'a str, Arc<dyn ImageViewAccess + Send + Sync>>,
    state: DrawState,
    pass_timers: Vec<Timer>,
    cmd_buf_timer: Timer,
    present_timer: Timer,
    acquire_timer: Timer,
    setup_timer: Timer,
}

enum DrawState {
    Uninitialized,
    Drawing {
        cmd_buf: AutoCommandBufferBuilder,
        pass_idx: usize,
        images: HashMap<String, Arc<dyn ImageViewAccess + Send + Sync>>,
        framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
        cur_dims: [u32; 2],
    },
}

// In the end all GPU programs come down to feeding a set of shaders some data
// and getting some data back. Vertex shaders take geometry and rasterize it,
// the output of which is stored in an image. If there are multiple outputs from
// the vertex shader, for example color and normal, 2 images will be created.
// The fragment shader then reads from these images to determine the final
// output color for each pixel on the screen. Compute shaders are another story.

// Passes specify which images the vertex shaders to write to and the fragment
// shaders read from. This does NOT mean textures! Data you want to feed your
// shaders from the CPU, whether in the form of buffers or images, should go in
// Object's custom_set field. The images listed in images_needed will be fed to
// the vertex shader of every object drawn.

// Often drawing a frame requires multiple vertex and fragment shaders operating
// in sequence. This what System is for.
pub struct Pass<'a> {
    pub name: &'a str,
    pub images_created_tags: Vec<&'a str>,
    pub images_needed_tags: Vec<&'a str>,
    pub render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
}

impl<'a> System<'a> {
    pub fn new(
        queue: Arc<Queue>,
        passes: Vec<Pass<'a>>,
        custom_images: HashMap<&'a str, Arc<dyn ImageViewAccess + Send + Sync>>,
        output_tag: &'a str,
    ) -> Self {
        let device = queue.device().clone();

        let pipeline_caches = pipe_caches_for_passes(device.clone(), &passes);
        let collection_cache = CollectionCache::new(device.clone());
        let pass_timers = passes.iter().map(|pass| Timer::new(pass.name)).collect();

        Self {
            passes,
            pipeline_caches,
            collection_cache,
            device,
            queue,
            output_tag,
            cached_images: None,
            custom_images,
            state: DrawState::Uninitialized,
            pass_timers,
            cmd_buf_timer: Timer::new("command buffer"),
            present_timer: Timer::new("present to window"),
            acquire_timer: Timer::new("acquire image from window"),
            setup_timer: Timer::new("frame setup"),
        }
    }

    pub fn start(&mut self, dest_image: Arc<dyn ImageViewAccess + Send + Sync>) {
        self.setup_timer.start();

        // all images will be created with the same dimensions as the
        // destination image. if you need to use an image with a different
        // resolution, use custom_images in System.
        let dimensions = [
            dest_image.dimensions().width(),
            dest_image.dimensions().height(),
        ];

        // create all images and framebuffers
        let mut images = self.get_images(dimensions);

        // replace destination image with the real one
        images.insert(self.output_tag.to_string(), dest_image);

        // use any custom images to replace existing ones
        for (tag, image) in self.custom_images.iter() {
            images.insert(tag.to_string(), image.clone());
        }

        let framebuffers = framebuffers_for_passes(images.clone(), &self.passes);

        // when you begin rendering, you automatically enter the first pass (for
        // which the first framebuffer is used)
        let first_framebuffer = framebuffers[0].clone();
        let first_render_pass = self.passes[0].render_pass.clone();

        let clear_values = clear_values_for_pass(first_render_pass);

        // create the command buffer and enter first render pass
        let cmd_buf_builder = AutoCommandBufferBuilder::primary_one_time_submit(
            self.device.clone(),
            self.queue.family(),
        )
        .unwrap()
        .begin_render_pass(first_framebuffer, false, clear_values.clone())
        .unwrap();

        self.state = DrawState::Drawing {
            cmd_buf: cmd_buf_builder,
            pass_idx: 0,
            images,
            framebuffers,
            // TODO: support passes with different dimensions
            cur_dims: dimensions,
        };

        self.setup_timer.stop();
        self.cmd_buf_timer.start();
        self.pass_timers[0].start();
    }

    pub fn start_window(&mut self, window: &mut Window) {
        self.acquire_timer.start();
        let swapchain_image = window.next_image();
        self.acquire_timer.stop();
        self.start(swapchain_image);
    }

    pub fn add_object<T: Drawcall>(&mut self, object: &T) {
        // we need to take ownership for a while
        let state = std::mem::replace(&mut self.state, DrawState::Uninitialized);
        match state {
            DrawState::Uninitialized => {
                panic!("You tried to render an object without calling begin_render first!")
            }
            DrawState::Drawing {
                mut cmd_buf,
                pass_idx,
                images,
                framebuffers,
                cur_dims,
            } => {
                // TODO: dynamic state is re-created for every object, shouldn't be
                let dynamic_state = if let Some(dynstate) = object.custom_dynstate() {
                    dynstate
                } else {
                    // TODO: this is another spot preventing passes with
                    // different dimensions
                    dynamic_state_for_dimensions(cur_dims)
                };

                let pipeline = self.pipeline_caches[pass_idx].get(object.pipe_spec());

                let mut collection = self.collection_cache.get(
                    object.pipe_spec(),
                    pipeline.clone(),
                    &self.passes[pass_idx],
                    &images,
                );

                let mut obj_collection =
                    object.collection();

                collection.append(&mut obj_collection);

                cmd_buf = cmd_buf
                    .draw_indexed(
                        pipeline,
                        &dynamic_state,
                        vec![object.vbuf()],
                        object.ibuf(),
                        collection,
                        (),
                    )
                    .expect(&format!(
                        "error building cmd buf, in pass {}",
                        self.passes[pass_idx].name
                    ));

                // give state a real value again
                self.state = DrawState::Drawing {
                    cmd_buf,
                    pass_idx,
                    images,
                    framebuffers,
                    cur_dims,
                }
            }
        }
    }

    pub fn next_pass(&mut self) {
        // again, temporarily take ownership
        let state = std::mem::replace(&mut self.state, DrawState::Uninitialized);
        match state {
            DrawState::Uninitialized => {
                panic!("Can't enter next pass without having begun rendering")
            }
            DrawState::Drawing {
                mut cmd_buf,
                mut pass_idx,
                images,
                framebuffers,
                cur_dims,
            } => {
                self.pass_timers[pass_idx].stop();
                pass_idx += 1;
                self.pass_timers[pass_idx].start();

                let framebuffer = framebuffers[pass_idx].clone();
                let render_pass = self.passes[pass_idx].render_pass.clone();
                let clear_values = clear_values_for_pass(render_pass);

                cmd_buf = cmd_buf
                    .end_render_pass()
                    .unwrap()
                    .begin_render_pass(framebuffer, false, clear_values)
                    .unwrap();

                // give state a real value again
                self.state = DrawState::Drawing {
                    cmd_buf,
                    pass_idx,
                    images,
                    framebuffers,
                    cur_dims,
                }
            }
        }
    }

    pub fn finish<F: GpuFuture + 'static>(&mut self, future: F) -> Box<dyn GpuFuture> {
        self.cmd_buf_timer.stop();

        let state = std::mem::replace(&mut self.state, DrawState::Uninitialized);

        let fut = match state {
            DrawState::Uninitialized => panic!("Can't finish render without having begun it"),
            DrawState::Drawing {
                cmd_buf, pass_idx, ..
            } => {
                self.pass_timers[pass_idx].stop();
                Box::new(
                    future
                        .then_execute(
                            self.queue.clone(),
                            cmd_buf.end_render_pass().unwrap().build().unwrap(),
                        )
                        .unwrap(),
                )
            }
        };

        fut
    }

    pub fn finish_to_window(&mut self, window: &mut Window) {
        self.present_timer.start();

        let swapchain_fut = window.get_future();
        let cmd_buf_fut = self.finish(swapchain_fut);
        window.present_future(cmd_buf_fut);

        self.present_timer.stop();
    }

    pub fn get_passes(&self) -> &[Pass] {
        &self.passes
    }

    pub fn print_stats(&self) {
        println!();

        self.cmd_buf_timer.print();
        self.present_timer.print();
        self.setup_timer.print();
        self.acquire_timer.print();
        self.pass_timers.iter().for_each(|timer| timer.print());

        println!();

        (0..self.passes.len()).for_each(|idx| {
            println!("Pipeline cache stats for pass {}:", self.passes[idx].name);
            self.pipeline_caches[idx].print_stats();
            println!();
            println!();
        });

        println!();
    }

    fn get_images(
        &mut self,
        dimensions: [u32; 2],
    ) -> HashMap<String, Arc<dyn ImageViewAccess + Send + Sync>> {
        // gets images to be drawn to either by using cached ones or creating
        // new ones

        // if there is a cache, make sure its dimensions are the same as what we want
        if let Some(cached) = &self.cached_images {
            let cached_vk_dims = cached
                .get(self.output_tag)
                .unwrap_or_else(|| {
                    panic!("Couldn't find output tag {} in cached_images", self.output_tag)
                })
                .dimensions();
            let cached_dimensions = [cached_vk_dims.width(), cached_vk_dims.height()];

            if cached_dimensions != dimensions {
                self.cached_images = None;
            }
        }

        if let Some(cached) = &self.cached_images {
            cached.clone()
        } else {
            let new = images_for_passes(self.device.clone(), dimensions, &self.passes);
            self.cached_images = Some(new.clone());
            new
        }
    }
}

fn create_image_for_desc(
    device: Arc<Device>,
    dimensions: [u32; 2],
    desc: AttachmentDescription,
) -> Arc<dyn ImageViewAccess + Send + Sync> {
    AttachmentImage::sampled_multisampled(device.clone(), dimensions, desc.samples, desc.format)
        .unwrap()
}

fn dynamic_state_for_dimensions(dimensions: [u32; 2]) -> DynamicState {
    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };

    DynamicState {
        line_width: None,
        viewports: Some(vec![viewport]),
        scissors: None,
    }
}

fn fb_from_images(
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    images: Vec<Arc<dyn ImageViewAccess + Send + Sync>>,
) -> Arc<dyn FramebufferAbstract + Send + Sync> {
    match images.len() {
        0 => panic!("You cannot create a framebuffer with 0 images!"),
        1 => Arc::new(
            Framebuffer::start(render_pass)
                .add(images[0].clone())
                .unwrap()
                .build()
                .unwrap(),
        ),
        2 => Arc::new(
            Framebuffer::start(render_pass)
                .add(images[0].clone())
                .unwrap()
                .add(images[1].clone())
                .unwrap()
                .build()
                .unwrap(),
        ),
        3 => Arc::new(
            Framebuffer::start(render_pass)
                .add(images[0].clone())
                .unwrap()
                .add(images[1].clone())
                .unwrap()
                .add(images[2].clone())
                .unwrap()
                .build()
                .unwrap(),
        ),
        4 => Arc::new(
            Framebuffer::start(render_pass)
                .add(images[0].clone())
                .unwrap()
                .add(images[1].clone())
                .unwrap()
                .add(images[2].clone())
                .unwrap()
                .add(images[3].clone())
                .unwrap()
                .build()
                .unwrap(),
        ),
        _ => panic!("Creating a framebuffer from more than 4 images is unsupported!"),
    }
}

fn images_for_passes<'a>(
    device: Arc<Device>,
    dimensions: [u32; 2],
    passes: &'a [Pass],
) -> HashMap<String, Arc<dyn ImageViewAccess + Send + Sync>> {
    // for now this ignores the fact that the output image is special and
    // provided from outside System if drawing to a window. any users of this
    // function should replace that image with the real one afterwards.

    let mut images = HashMap::new();
    for pass in passes.iter() {
        let num_tags = pass.images_created_tags.len();
        let num_rpass_images = pass.render_pass.num_attachments();

        assert!(num_rpass_images == num_tags, "There are a different number of image tags ({}) than images in the render pass ({}) for pass {}!
This means the image description for each tag cannot be determined from the render pass!",
                num_tags,
                num_rpass_images,
                pass.name,
        );

        for (image_idx, &image_tag) in pass.images_created_tags.iter().enumerate() {
            let desc = pass
                .render_pass
                .attachment_desc(image_idx)
                .expect(&format!(
                    "Couldn't get the attachment description when creating images for passes.

Dimensions we're supposed to be creating for: {:?}
Image index we tried to create: {}
Image tag: {}",
                    dimensions,
                    image_idx,
                    image_tag,
                ));

            // FIXME: yeah this needs a better solution
            let image = if image_tag.contains("lowres") {
                create_image_for_desc(device.clone(), [512, 512], desc)
            } else {
                create_image_for_desc(device.clone(), dimensions, desc)
            };

            images.insert(image_tag.to_string(), image);
        }
    }

    images
}

fn framebuffers_for_passes<'a>(
    images: HashMap<String, Arc<dyn ImageViewAccess + Send + Sync>>,
    passes: &'a [Pass],
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let mut framebuffers = vec![];

    for pass in passes.iter() {
        let images_tags_created = &pass.images_created_tags;
        let images = images_tags_created
            .iter()
            .map(|tag| {
                images
                    .get(&tag.to_string())
                    .expect("Couldn't get image when creating framebuffers for passes")
                    .clone()
            })
            .collect();

        let framebuffer = fb_from_images(pass.render_pass.clone(), images);
        framebuffers.push(framebuffer);
    }

    framebuffers
}

fn pipe_caches_for_passes(device: Arc<Device>, passes: &[Pass]) -> Vec<PipelineCache> {
    passes
        .iter()
        .map(|pass| PipelineCache::new(device.clone(), pass.render_pass.clone()))
        .collect()
}
