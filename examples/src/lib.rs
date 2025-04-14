use render_engine::input::{FrameInfo, get_elapsed};
use render_engine::utils::upload_data;
use render_engine::{Buffer, Device};
use render_engine::collection::Data;

use nalgebra_glm::*;

use std::path::PathBuf;
use std::convert::From;

pub mod mesh;

pub fn relative_path(local_path: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), local_path].iter().collect()
}

#[derive(Clone, Copy)]
pub struct Matrix4([[f32; 4]; 4]);
impl Data for Matrix4 {}

impl From<[[f32; 4]; 4]> for Matrix4 {
    fn from(item: [[f32; 4]; 4]) -> Self {
        Self(item)
    }
}

impl From<Mat4> for Matrix4 {
    fn from(item: Mat4) -> Self {
        let data: [[f32; 4]; 4] = item.into();
        Self(data)
    }
}

#[derive(Clone)]
pub struct OrbitCamera {
    pub center_position: Vec3,
    pub front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,
    // pitch and yaw are in radians
    pub pitch: f32,
    pub yaw: f32,
    pub orbit_distance: f32,
    mouse_sens: f32,
    view_mat: CameraMatrix,
    proj_mat: CameraMatrix,
}

// TODO: builders for changing fov, perspective, orbit dist, etc.
impl OrbitCamera {
    pub fn default() -> Self {
        let center_position = vec3(0.0, 0.0, 0.0);
        let pitch: f32 = 0.0;
        let yaw: f32 = std::f32::consts::PI / 2.0;
        let front = normalize(&vec3(
            pitch.cos() * yaw.cos(),
            pitch.sin(),
            pitch.cos() * yaw.sin(),
        ));
        let right = vec3(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);
        let world_up = vec3(0.0, 1.0, 0.0);
        let mouse_sens = 0.0007;
        let orbit_distance = 20.0;

        let view_mat: CameraMatrix = Mat4::identity().into();
        let proj_mat: CameraMatrix = Mat4::identity().into();

        Self {
            center_position,
            front,
            up,
            right,
            world_up,
            pitch,
            yaw,
            orbit_distance,
            mouse_sens,
            view_mat,
            proj_mat,
        }
    }

    pub fn update(&mut self, frame_info: FrameInfo) {
        // check for scroll wheel
        let scroll: f32 = frame_info
            .all_events
            .iter()
            .map(|ev| match ev {
                winit::Event::WindowEvent {
                    event:
                        winit::WindowEvent::MouseWheel {
                            delta: winit::MouseScrollDelta::LineDelta(_, y),
                            ..
                        },
                    ..
                } => *y,
                _ => 0.0,
            })
            .sum();

        self.orbit_distance += scroll;

        // TODO: a lot of the stuff stored in OrbitCamera doesn't need to be
        // stored across frames
        let x = frame_info.mouse_movement[0];
        let y = frame_info.mouse_movement[1];

        self.pitch -= y * self.mouse_sens;
        self.yaw += x * self.mouse_sens;
        let halfpi = std::f32::consts::PI / 2.0;
        let margin = 0.01;
        let max_pitch = halfpi - margin;

        if self.pitch > max_pitch {
            self.pitch = max_pitch;
        } else if self.pitch < -max_pitch {
            self.pitch = -max_pitch;
        }

        // recompute front vector
        self.front = normalize(&vec3(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        ));

        self.right = normalize(&Vec3::cross(&self.front, &self.world_up));

        // recompute view and projection matrices
        let farther_front = self.front * self.orbit_distance;
        self.view_mat = look_at(
            &(self.center_position + farther_front),
            &self.center_position,
            &self.up,
        )
        .into();

        let dims = frame_info.dimensions;
        let aspect_ratio = (dims[0] as f32) / (dims[1] as f32);
        // TODO: idk why i have to flip it vertically
        self.proj_mat = scale(
            &perspective(
                aspect_ratio,
                // fov
                1.0,
                // near
                1.0,
                // far
                10_000.,
            ),
            &vec3(1.0, -1.0, 1.0),
        )
        .into();
    }

    pub fn get_data(&self) -> CameraData {
        CameraData {
            view: self.view_mat,
            proj: self.proj_mat,
            pos: (self.front * self.orbit_distance).into(),
        }
    }
}

pub struct FlyCamera {
    pub position: Vec3,
    pub front: Vec3,
    up: Vec3,
    right: Vec3,
    world_up: Vec3,
    // pitch and yaw are in radians
    pub pitch: f32,
    pub yaw: f32,
    movement_speed: f32,
    mouse_sens: f32,
    view_mat: CameraMatrix,
    proj_mat: CameraMatrix,
}

impl FlyCamera {
    pub fn default() -> Self {
        let position = vec3(0.0, 0.0, 0.0);
        let pitch: f32 = 0.0;
        let yaw: f32 = std::f32::consts::PI / 2.0;
        let front = normalize(&vec3(
            pitch.cos() * yaw.cos(),
            pitch.sin(),
            pitch.cos() * yaw.sin(),
        ));
        let right = vec3(0.0, 0.0, 0.0);
        let up = vec3(0.0, 1.0, 0.0);
        let world_up = vec3(0.0, 1.0, 0.0);
        let movement_speed = 20.0;
        let mouse_sens = 0.0007;

        Self {
            position,
            front,
            up,
            right,
            world_up,
            pitch,
            yaw,
            movement_speed,
            mouse_sens,
            view_mat: Mat4::identity().into(),
            proj_mat: Mat4::identity().into(),
        }
    }

    pub fn move_forward(&mut self, delta: f32) {
        self.position += self.front * self.movement_speed * delta;
    }

    pub fn move_backward(&mut self, delta: f32) {
        self.position -= self.front * self.movement_speed * delta;
    }

    pub fn move_left(&mut self, delta: f32) {
        self.position -= self.right * self.movement_speed * delta;
    }

    pub fn move_right(&mut self, delta: f32) {
        self.position += self.right * self.movement_speed * delta;
    }

    pub fn update(&mut self, frame_info: FrameInfo) {
        let x = frame_info.mouse_movement[0];
        let y = frame_info.mouse_movement[1];

        self.pitch -= y * self.mouse_sens;
        self.yaw += x * self.mouse_sens;
        let halfpi = std::f32::consts::PI / 2.0;
        let margin = 0.01;
        let max_pitch = halfpi - margin;

        if self.pitch > max_pitch {
            self.pitch = max_pitch;
        } else if self.pitch < -max_pitch {
            self.pitch = -max_pitch;
        }

        // move if keys are down
        let move_dist = if frame_info.keys_down.x {
            frame_info.delta * 3.0
        } else {
            frame_info.delta
        };
        if frame_info.keys_down.w {
            self.move_forward(move_dist);
        }
        if frame_info.keys_down.a {
            self.move_left(move_dist);
        }
        if frame_info.keys_down.s {
            self.move_backward(move_dist);
        }
        if frame_info.keys_down.d {
            self.move_right(move_dist);
        }

        // update front and right
        self.front = normalize(&vec3(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        ));

        self.right = normalize(&Vec3::cross(&self.front, &self.world_up));

        self.view_mat = look_at(&self.position, &(self.position + self.front), &self.up).into();

        let dims = frame_info.dimensions;
        let aspect_ratio = (dims[0] as f32) / (dims[1] as f32);
        // TODO: idk why i have to flip it vertically
        self.proj_mat = scale(
            &perspective(
                aspect_ratio,
                // fov
                1.0,
                // near
                1.0,
                // far
                10_000.,
            ),
            &vec3(1.0, -1.0, 1.0),
        )
        .into();
    }

    pub fn get_data(&self) -> CameraData {
        CameraData {
            view: self.view_mat,
            proj: self.proj_mat,
            pos: self.position.into(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct CameraData {
    view: CameraMatrix,
    proj: CameraMatrix,
    pos: [f32; 3],
}
impl Data for CameraData {}

pub type CameraMatrix = [[f32; 4]; 4];

#[allow(dead_code)]
pub struct Light {
    direction: [f32; 4],
    power: f32,
}

pub struct MovingLight {
    start_time: std::time::Instant,
}

impl MovingLight {
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
        }
    }

    pub fn get_buffer(&self, device: Device) -> Buffer {
        let time = get_elapsed(self.start_time) / 4.0;
        let data = Light {
            direction: [time.sin(), 2.0, time.cos(), 0.0],
            power: 1.0,
        };

        upload_data(device, data)
    }
}
