use vulkano::device::Device;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::GraphicsPipelineAbstract;

use std::path::PathBuf;
use std::sync::Arc;

use crate::input::get_elapsed;
use crate::mesh::VertexTypeAbstract;
use crate::shaders::ShaderSystem;

// pipeline caches are specific to a single render pass.
pub struct PipelineCache {
    // TODO: switch to a hashmap
    c_pipes: Vec<CachedPipeline>,
    device: Arc<Device>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    stats: CacheStats,
}

struct CachedPipeline {
    spec: PipelineSpec,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

#[derive(Default)]
struct CacheStats {
    hits: u32,
    misses: u32,
    gen_times: Vec<f32>,
}

// TODO: rename to pipeline_prototype or something more accurate.
pub struct PipelineSpec {
    pub vs_path: PathBuf,
    pub fs_path: PathBuf,
    pub fill_type: PrimitiveTopology,
    pub read_depth: bool,
    pub write_depth: bool,
    pub vtype: Arc<dyn VertexTypeAbstract>
}

impl PipelineCache {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Self {
        Self {
            c_pipes: vec![],
            device,
            render_pass,
            stats: CacheStats::default(),
        }
    }

    pub fn get(&mut self, spec: &PipelineSpec) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        let mut pipeline = None;

        // first search through cached pipelines to see if we have one with matching spec
        for c_pipe in self.c_pipes.iter() {
            // TODO: yooooooo fix this fix this fix this
            if c_pipe.spec == *spec {
                pipeline = Some(c_pipe.pipeline.clone());
                self.stats.hits += 1;
            }
        }

        match pipeline {
            Some(pipeline) => pipeline,
            None => {
                self.stats.misses += 1;
                let start_time = std::time::Instant::now();

                let pipeline = spec.concrete(self.device.clone(), self.render_pass.clone());
                let c_pipe = CachedPipeline {
                    spec: spec.clone(),
                    pipeline: pipeline.clone(),
                };

                self.c_pipes.push(c_pipe);

                self.stats.gen_times.push(get_elapsed(start_time));

                pipeline
            }
        }
    }

    pub fn print_stats(&self) {
        let avg: f32 =
            self.stats.gen_times.iter().sum::<f32>() / (self.stats.gen_times.len() as f32) * 1_000.0;
        let percent =
            (self.stats.hits as f32) / ((self.stats.hits + self.stats.misses) as f32) * 100.0;
        println!(
            "Hits: {}, misses: {}, {}%, avg. time taken to gen pipeline: {}ms",
            self.stats.hits, self.stats.misses, percent, avg
        );
    }
}

impl PipelineSpec {
    pub fn concrete(&self, device: Arc<Device>, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        let shader_sys =
            ShaderSystem::load_from_file(device.clone(), &self.vs_path, &self.fs_path);

        self.vtype.create_pipeline(
            device,
            shader_sys,
            self.fill_type,
            render_pass,
            self.read_depth,
            self.write_depth,
        )
    }
}

impl PartialEq for PipelineSpec {
    fn eq(&self, other: &Self) -> bool {
        // TODO: compare vtype too
        self.vs_path == other.vs_path
            && self.fs_path == other.fs_path
            && self.fill_type == other.fill_type
            && self.read_depth == other.read_depth
            && self.write_depth == other.write_depth
    }
}

impl Clone for PipelineSpec {
    fn clone(&self) -> Self {
        PipelineSpec {
            vs_path: self.vs_path.clone(),
            fs_path: self.fs_path.clone(),
            fill_type: self.fill_type,
            read_depth: self.read_depth,
            write_depth: self.write_depth,
            vtype: self.vtype.clone(),
        }
    }
}
