struct VertexInput{
    @location(0) position: vec3<f32>,
}

struct VertexOutput{
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct ViewProj{
    view: mat4x4<f32>,
    proj: mat4x4<f32>,

}

var<push_constant> view_proj:  ViewProj;

fn ray_sphere_intersection(ray_origin: vec3<f32>, ray_direction: vec3<f32>, sphere_center: vec3<f32>, sphere_radius: f32) -> f32{
    let oc = ray_origin - sphere_center;
    let a = dot(ray_direction, ray_direction);
    let b = 2.0 * dot(oc, ray_direction);
    let c = dot(oc, oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;
    if(discriminant < 0.0){
        return -1.0;
    }else{
        return (-b - sqrt(discriminant)) / (2.0 * a);
    }
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32,) -> VertexOutput {
    var id = in_vertex_index;
    let _uv = vec2<u32>( ((id << 1u) & 2u), in_vertex_index & 2u);
    let uv = vec2<f32>(_uv);

    var out: VertexOutput;
    out.clip_pos = vec4<f32>(uv * 2.0f + -1.0f, 0.0f, 1.0f);
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    //view space

    //get aspect ratio from view proj matrix
    let aspect_ratio = view_proj.proj[1][1] / view_proj.proj[0][0];

    //ray from camera
    let uv = vec2<f32>(in.uv.x * aspect_ratio, in.uv.y);

    let ray_direction = vec3<f32>((in.uv - 0.5), -1.0);

    let sun_pos = vec3<f32>(0.0, 0.0, 0.0);
    let sun_pos_view = view_proj.view * vec4<f32>(sun_pos, 1.0);

    let d = ray_sphere_intersection(vec3<f32>(0.0, 0.0, 0.0) ,  ray_direction , sun_pos_view.xyz, 0.1);
    if(d > 0.0){
        return vec4<f32>(vec3<f32>(d), 1.0);
    }
    else{
        discard;
    }
}