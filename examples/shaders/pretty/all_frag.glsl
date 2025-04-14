#version 450

layout(location = 0) in vec2 v_tex_coord;
layout(location = 1) in vec3 tan_light_pos;
layout(location = 2) in vec3 tan_cam_pos;
layout(location = 3) in vec3 tan_frag_pos;
layout(location = 4) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D shadow_map;
layout(set = 1, binding = 0) uniform Material {
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
  vec3 shininess;
  vec3 use_texture;
} material;

layout(set = 1, binding = 1) uniform Model {
  mat4 model;
} model;

layout(set = 2, binding = 0) uniform sampler2D diffuse_map;
layout(set = 2, binding = 1) uniform sampler2D specular_map;
layout(set = 2, binding = 2) uniform sampler2D normal_map;

layout(set = 3, binding = 0) uniform Camera {
  mat4 view;
  mat4 proj;
  vec3 pos;
} camera;

layout(set = 3, binding = 1) uniform Light {
  vec3 position;
  vec3 strength; // vec3 really means float, idk why it doesn't work
} light;

float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

// taken from: http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 Uncharted2Tonemap(vec3 x)
{
  return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

// cube faces +x, -x, +y, -y, +z, -z in a row
// taken from: http://blue2rgb.sydneyzh.com/rendering-dynamic-cube-maps-for-omni-light-shadows-with-vulkan-api.html
vec2 l_to_shadow_map_uv(vec3 v) {
  float face_index;
  vec3 v_abs = abs(v);
  float ma;
  vec2 uv;
  if(v_abs.z >= v_abs.x && v_abs.z >= v_abs.y)
    {
      face_index = v.z < 0.0 ? 5.0 : 4.0;
      ma = 0.5 / v_abs.z;
      uv = vec2(v.z < 0.0 ? -v.x : v.x, -v.y);
    }
  else if(v_abs.y >= v_abs.x)
    {
      face_index = v.y < 0.0 ? 3.0 : 2.0;
      ma = 0.5 / v_abs.y;
      uv = vec2(v.x, v.y < 0.0 ? -v.z : v.z);
    }
  else
    {
      face_index = v.x < 0.0 ? 1.0 : 0.0;
      ma = 0.5 / v_abs.x;
      uv = vec2(v.x < 0.0 ? v.z : -v.z, -v.y);
    }
  uv = uv * ma + 0.5;
  uv = uv * 0.9921875 + 0.00390625;
  uv.x = (uv.x + face_index) / 6.f;
  return uv;
}

float shadowedness() {
  vec3 light_dir = normalize(v_pos - light.position);
  vec2 coords = l_to_shadow_map_uv(light_dir);
  float sample_dist = texture(shadow_map, coords).r * 250.0;

  float frag_dist = length(v_pos - light.position);
  float bias = 0.05;

  // idk why i have to invert it
  float difference = abs(sample_dist - frag_dist);

  return clamp(difference, 0.0, 1.0);
  /* return !(sample_dist + bias > frag_dist); */
}

void main() {
  // only use the texture if we should
  vec4 tex_diffuse = material.use_texture.r > 0.5 ? texture(diffuse_map, v_tex_coord) : vec4(material.diffuse, 1.0);

  // doesn't play nice with depth prepass
  /*
  if (tex_diffuse.a < 0.5) {
    discard;
  }
  */

  vec3 tex_specular = texture(specular_map, v_tex_coord).rgb;

  vec3 normal = texture(normal_map, v_tex_coord).rgb * 2.0 - 1.0;

  // ambient
  vec3 ambient = tex_diffuse.rgb * 0.01;

  // diffuse
  vec3 light_dir = normalize(tan_light_pos - tan_frag_pos);

  float diff = max(dot(normal, light_dir), 0.0);
  vec3 diffuse = diff * tex_diffuse.rgb;

  // specular
  vec3 view_dir = normalize(tan_cam_pos - tan_frag_pos);
  vec3 halfway_dir = normalize(light_dir + view_dir);
  float spec = pow(max(dot(normal, halfway_dir), 0.0), 32.0);
  vec3 specular = vec3(clamp(0.2 * spec, 0.0, 0.5));

  // result
  float dist = length(tan_light_pos - tan_frag_pos);
  float shadow = shadowedness();

  vec3 result = ambient + (1.0 - shadow) * (diffuse + specular) * light.strength.r / (dist * dist / 2000.0);

  // uncharted 2 tone mapping
  result *= 16;
  float exposure_bias = 2.0;
  vec3 curr = Uncharted2Tonemap(exposure_bias * result);

  vec3 corrected = pow(curr, vec3(1/2.2));

  f_color = vec4(corrected, 1.0);
}
