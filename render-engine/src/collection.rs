/*
There are two types of data that can be used in a collection (data passed to
shaders): images and structs. Structs must implement the Data trait to be
uploaded to the GPU, which just means implementing Send, Sync, Clone and being
'static.

The SetUpload trait is implemented for any* tuple of images and structs that
implement Data. For example, it is implemented for (Image, Data, Image) and
(Data, Data, Data) and (Image,) and (Data, Image,) and so on.

*: any tuple up to size 3. Sorry.

These tuples should represent a set within a collection that will be used in a
shader. SetUpload requires implementing upload, which uploads the data to the
GPU and returns an Arc<dyn DescriptorSet + Send + Sync>.

The Set struct contains some data, a cached set, and the resources required to
re-upload the set in case one of its elements changes. This is real handy,
because it means you can initialize the set once with all the annoying data
necessary to upload it (the pipeline) and easily re-upload it and change the
underlying data.

let mut set = Set::new(
    (some_struct,),
    device,
    pipeline,
    0      // set idx
);
set.data.0 = updated_struct;
set.upload(device);

Ta-da! Now to collections. Collection is a trait implemented for all* tuples of
sets that allows converting them into Vec<Arc<DescriptorSet>>, which is most
concrete form of a collection: it is the type taken by draw and draw_indexed
when creating the command buffer. Collection requires the get() function, which
returns a Vec<Arc<DescriptorSet>>.

*: any tuple up to size 4.

So that's it: how you can go from tuples of images and arbitrary structs to a
type that can be used in draw and draw_indexed. How magnificently mediocre.
 */

use vulkano::descriptor::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::device::Device;
use vulkano::image::ImageViewAccess;
use vulkano::pipeline::GraphicsPipelineAbstract;

use crate::utils::{upload_data, default_sampler};

use std::sync::Arc;

pub trait Collection {
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>>;
}

impl Collection for () {
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        vec![]
    }
}

impl<T: SetUpload> Collection for (Set<T>,) {
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        vec![self.0.get()]
    }
}

impl<T1: SetUpload, T2: SetUpload> Collection for (Set<T1>, Set<T2>) {
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        vec![self.0.get(), self.1.get()]
    }
}

impl<T1: SetUpload, T2: SetUpload, T3: SetUpload> Collection for (Set<T1>, Set<T2>, Set<T3>) {
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        vec![self.0.get(), self.1.get(), self.2.get()]
    }
}

impl<T1: SetUpload, T2: SetUpload, T3: SetUpload, T4: SetUpload> Collection
    for (Set<T1>, Set<T2>, Set<T3>, Set<T4>)
{
    fn get(&self) -> Vec<Arc<dyn DescriptorSet + Send + Sync>> {
        vec![self.0.get(), self.1.get(), self.2.get(), self.3.get()]
    }
}

/*
CollectionData
 */

pub trait CollectionData {
    type Sets: Collection;

    fn create_sets(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx_offset: usize,
    ) -> Self::Sets;
}

impl CollectionData for () {
    type Sets = ();

    fn create_sets(
        &self,
        _device: Arc<Device>,
        _pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        _set_idx_offset: usize,
    ) -> Self::Sets {
    }
}

impl<T1: SetUpload> CollectionData for (T1,) {
    type Sets = (Set<T1>,);

    fn create_sets(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx_offset: usize,
    ) -> Self::Sets {
        let set1 = Set::new(self.0.clone(), device, pipeline, set_idx_offset);

        (set1,)
    }
}

impl<T1: SetUpload, T2: SetUpload> CollectionData for (T1, T2) {
    type Sets = (Set<T1>, Set<T2>);

    fn create_sets(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx_offset: usize,
    ) -> Self::Sets {
        let set1 = Set::new(
            self.0.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset,
        );
        let set2 = Set::new(
            self.1.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 1,
        );

        (set1, set2)
    }
}

impl<T1: SetUpload, T2: SetUpload, T3: SetUpload> CollectionData for (T1, T2, T3) {
    type Sets = (Set<T1>, Set<T2>, Set<T3>);

    fn create_sets(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx_offset: usize,
    ) -> Self::Sets {
        let set1 = Set::new(
            self.0.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset,
        );
        let set2 = Set::new(
            self.1.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 1,
        );
        let set3 = Set::new(
            self.2.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 2,
        );

        (set1, set2, set3)
    }
}

impl<T1: SetUpload, T2: SetUpload, T3: SetUpload, T4: SetUpload> CollectionData
    for (T1, T2, T3, T4)
{
    type Sets = (Set<T1>, Set<T2>, Set<T3>, Set<T4>);

    fn create_sets(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx_offset: usize,
    ) -> Self::Sets {
        let set1 = Set::new(
            self.0.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset,
        );
        let set2 = Set::new(
            self.1.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 1,
        );
        let set3 = Set::new(
            self.2.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 2,
        );
        let set4 = Set::new(
            self.3.clone(),
            device.clone(),
            pipeline.clone(),
            set_idx_offset + 3,
        );

        (set1, set2, set3, set4)
    }
}

/*
Set
 */
#[derive(Clone)]
pub struct Set<T: SetUpload> {
    pub data: T,
    cached: Arc<dyn DescriptorSet + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    set_idx: usize,
}

impl<T: SetUpload> Set<T> {
    pub fn new(
        data: T,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Self {
        // creates a new set and immediately uploads the data to the GPU
        let gpu_data = data.upload(device, pipeline.clone(), set_idx);
        Self {
            data,
            cached: gpu_data,
            pipeline,
            set_idx,
        }
    }

    pub fn get(&self) -> Arc<dyn DescriptorSet + Send + Sync> {
        self.cached.clone()
    }

    pub fn upload(&mut self, device: Arc<Device>) {
        self.cached = self.data.upload(device, self.pipeline.clone(), self.set_idx);
    }
}

pub trait SetUpload: Clone {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync>;
}

// length 1
impl<T: Data> SetUpload for (T,) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let buffer = upload_data(device.clone(), self.0.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl SetUpload for (Image,) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler)
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

// length 2
impl<T1: Data, T2: Data> SetUpload for (T1, T2) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let buffer1 = upload_data(device.clone(), self.0.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T: Data> SetUpload for (Image, T) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler)
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T: Data> SetUpload for (T, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer1 = upload_data(device.clone(), self.0.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.1.clone(), sampler)
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl SetUpload for (Image, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler.clone())
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_sampled_image(self.1.clone(), sampler)
                .expect(&format!("Panic adding 2nd image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

// length 3
impl<T1: Data, T2: Data, T3: Data> SetUpload for (T1, T2, T3) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let buffer1 = upload_data(device.clone(), self.0.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());
        let buffer3 = upload_data(device.clone(), self.2.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .add_buffer(buffer3)
                .expect(&format!("Panic adding 3rd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T1: Data, T2: Data> SetUpload for (Image, T1, T2) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());
        let buffer3 = upload_data(device.clone(), self.2.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler)
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .add_buffer(buffer3)
                .expect(&format!("Panic adding 3rd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T1: Data, T2: Data> SetUpload for (T1, Image, T2) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer1 = upload_data(device.clone(), self.0.clone());
        let buffer3 = upload_data(device.clone(), self.2.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .add_sampled_image(self.1.clone(), sampler)
                .expect(&format!("Panic adding 2nd image at set idx {}", set_idx))
                .add_buffer(buffer3)
                .expect(&format!("Panic adding 3rd buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T1: Data, T2: Data> SetUpload for (T1, T2, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer1 = upload_data(device.clone(), self.0.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .add_sampled_image(self.2.clone(), sampler)
                .expect(&format!("Panic adding 3rd image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T: Data> SetUpload for (T, Image, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer1 = upload_data(device.clone(), self.0.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_buffer(buffer1)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .add_sampled_image(self.1.clone(), sampler.clone())
                .expect(&format!("Panic adding 2nd image at set idx {}", set_idx))
                .add_sampled_image(self.2.clone(), sampler)
                .expect(&format!("Panic adding 3rd image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T: Data> SetUpload for (Image, T, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer2 = upload_data(device.clone(), self.1.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler.clone())
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_buffer(buffer2)
                .expect(&format!("Panic adding 2nd buffer at set idx {}", set_idx))
                .add_sampled_image(self.2.clone(), sampler)
                .expect(&format!("Panic adding 3rd image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl<T: Data> SetUpload for (Image, Image, T) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());
        let buffer3 = upload_data(device.clone(), self.2.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler.clone())
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_sampled_image(self.1.clone(), sampler)
                .expect(&format!("Panic adding 2nd image at set idx {}", set_idx))
                .add_buffer(buffer3)
                .expect(&format!("Panic adding 1st buffer at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

impl SetUpload for (Image, Image, Image) {
    fn upload(
        &self,
        device: Arc<Device>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
        set_idx: usize,
    ) -> Arc<dyn DescriptorSet + Send + Sync> {
        let sampler = default_sampler(device.clone());

        Arc::new(
            PersistentDescriptorSet::start(pipeline, set_idx)
                .add_sampled_image(self.0.clone(), sampler.clone())
                .expect(&format!("Panic adding 1st image at set idx {}", set_idx))
                .add_sampled_image(self.1.clone(), sampler.clone())
                .expect(&format!("Panic adding 2nd image at set idx {}", set_idx))
                .add_sampled_image(self.2.clone(), sampler)
                .expect(&format!("Panic adding 3rd image at set idx {}", set_idx))
                .build()
                .expect(&format!("Panic finalizing set at set idx {}", set_idx)),
        )
    }
}

pub type Image = Arc<dyn ImageViewAccess + Send + Sync>;

pub trait Data: Send + Sync + Clone + 'static {}
