#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 v_pos;

layout(set = 0, binding = 0) uniform sampler2D depth_map;

void main() {
  v_pos = position * 0.5 + 0.5;
  gl_Position = vec4(position, 1.0, 1.0);
}
