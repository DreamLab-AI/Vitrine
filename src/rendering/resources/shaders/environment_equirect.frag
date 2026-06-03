#version 430 core

in vec2 TexCoord;

uniform sampler2D u_environment;
uniform mat3 u_camera_to_world;
uniform vec2 u_viewport_size;
uniform vec4 u_intrinsics;
uniform float u_environment_exposure;
uniform float u_rotation_radians;
uniform bool u_equirectangular_view;

layout(location = 0) out vec4 frag_color;

const float PI = 3.14159265358979323846;

vec3 rotate_y(vec3 v, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(
        c * v.x + s * v.z,
        v.y,
        -s * v.x + c * v.z
    );
}

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 ray_direction_local() {
    if (u_equirectangular_view) {
        float lon = (TexCoord.x - 0.5) * (2.0 * PI);
        // TexCoord.y is bottom-to-top (OpenGL convention); latitude should increase towards +Y.
        float lat = (TexCoord.y - 0.5) * PI;
        float cos_lat = cos(lat);
        return normalize(vec3(
            sin(lon) * cos_lat,
            sin(lat),
            -cos(lon) * cos_lat
        ));
    }

    vec2 pixel = TexCoord * u_viewport_size;
    vec2 centered = vec2(
        (pixel.x - u_intrinsics.z) / max(u_intrinsics.x, 1e-6),
        // pixel.y grows upwards here (TexCoord), so +Y should be above the principal point.
        (pixel.y - u_intrinsics.w) / max(u_intrinsics.y, 1e-6)
    );
    return normalize(vec3(centered, -1.0));
}

vec2 environment_uv(vec3 dir) {
    float longitude = atan(dir.x, -dir.z);
    float latitude = asin(clamp(dir.y, -1.0, 1.0));
    return vec2(
        longitude / (2.0 * PI) + 0.5,
        0.5 - latitude / PI
    );
}

void main() {
    vec3 local_dir = ray_direction_local();
    vec3 world_dir = normalize(u_camera_to_world * local_dir);
    world_dir = normalize(rotate_y(world_dir, u_rotation_radians));

    vec2 uv = environment_uv(world_dir);
    // Avoid discontinuous derivatives at the wrap seam (u=0/1) by forcing a fixed LOD.
    uv.x = fract(uv.x);
    uv.y = clamp(uv.y, 0.0, 1.0);
    vec3 color = textureLod(u_environment, uv, 0.0).rgb;
    color *= exp2(u_environment_exposure);
    color = aces_tonemap(color);
    color = pow(color, vec3(1.0 / 2.2));

    frag_color = vec4(color, 1.0);
}
