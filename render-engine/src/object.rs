use vulkano::buffer::{BufferAccess, ImmutableBuffer};
use vulkano::command_buffer::DynamicState;
use vulkano::descriptor::DescriptorSet;
use vulkano::device::Queue;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::pipeline::input_assembly::PrimitiveTopology;

use crate::collection::{Collection, CollectionData};
use crate::mesh::{Mesh, MeshAbstract, Vertex, VertexType};
use crate::pipeline_cache::PipelineSpec;

use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct Object<C: Collection> {
    pub pipeline_spec: PipelineSpec,
    pub vbuf: Arc<dyn BufferAccess + Send + Sync>,
    pub ibuf: Arc<ImmutableBuffer<[u32]>>,
    pub collection: C,
    pub custom_dynamic_state: Option<DynamicState>,
}

pub trait Drawcall {
    fn pipe_spec(&self) -> &PipelineSpec;
    fn vbuf(&self) -> Arc<dyn BufferAccess + Send + Sync>;
    fn ibuf(&self) -> Arc<ImmutableBuffer<[u32]>>;
    fn collection(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>>;
    fn custom_dynstate(&self) -> Option<DynamicState>;
}

impl<C: Collection> Drawcall for Object<C> {
    fn pipe_spec(&self) -> &PipelineSpec {
        &self.pipeline_spec
    }

    fn vbuf(&self) -> Arc<dyn BufferAccess + Send + Sync> {
        self.vbuf.clone()
    }

    fn ibuf(&self) -> Arc<ImmutableBuffer<[u32]>> {
        self.ibuf.clone()
    }

    fn collection(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        self.collection.get()
    }

    fn custom_dynstate(&self) -> Option<DynamicState> {
        self.custom_dynamic_state.clone()
    }
}

impl<'a, C: Collection> Drawcall for &'a Object<C> {
    fn pipe_spec(&self) -> &PipelineSpec {
        &self.pipeline_spec
    }

    fn vbuf(&self) -> Arc<dyn BufferAccess + Send + Sync> {
        self.vbuf.clone()
    }

    fn ibuf(&self) -> Arc<ImmutableBuffer<[u32]>> {
        self.ibuf.clone()
    }

    fn collection(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        self.collection.get()
    }

    fn custom_dynstate(&self) -> Option<DynamicState> {
        self.custom_dynamic_state.clone()
    }
}

#[derive(Clone)]
pub struct ObjectPrototype<V: Vertex, D: CollectionData> {
    pub vs_path: PathBuf,
    pub fs_path: PathBuf,
    pub fill_type: PrimitiveTopology,
    pub read_depth: bool,
    pub write_depth: bool,
    pub mesh: Mesh<V>,
    pub collection: D,
    pub custom_dynamic_state: Option<DynamicState>,
}

impl<V: Vertex, D: CollectionData + 'static> ObjectPrototype<V, D> {
    pub fn build(
        self,
        queue: Arc<Queue>,
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        set_start_idx: usize,
    ) -> Object<D::Sets> {
        let vbuf = self.mesh.get_vbuf(queue.clone());
        let ibuf = self.mesh.get_ibuf(queue.clone());

        let pipeline_spec = PipelineSpec {
            vs_path: self.vs_path,
            fs_path: self.fs_path,
            fill_type: self.fill_type,
            read_depth: self.read_depth,
            write_depth: self.write_depth,
            vtype: VertexType::<V>::new(),
        };
        println!("Before pipeline");
        let pipeline = pipeline_spec.concrete(queue.device().clone(), render_pass);

        println!("Before create_sets");
        let collection = self.collection.create_sets(queue.device().clone(), pipeline,
            set_start_idx);

        Object {
            pipeline_spec,
            vbuf,
            ibuf,
            collection,
            custom_dynamic_state: self.custom_dynamic_state,
        }
    }
}
