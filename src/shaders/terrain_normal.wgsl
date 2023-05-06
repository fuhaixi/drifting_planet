struct ViewProj {
	view: mat4x4<f32>,
    proj: mat4x4<f32>,
}


struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) tex_coords: vec2<f32>,
	@location(2) normal: vec3<f32>,
}

struct InstanceInput {
	@location(5) model_matrix_0: vec4<f32>,
	@location(6) model_matrix_1: vec4<f32>,
	@location(7) model_matrix_2: vec4<f32>,
	@location(8) model_matrix_3: vec4<f32>,

}



struct SunLight {
	dir: vec3<f32>,
	color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> light: SunLight;


struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
	@location(0) tex_coords: vec2<f32>,
	@location(1) world_normal: vec3<f32>,
	@location(2) world_position: vec3<f32>,
}



var<push_constant> view_proj: ViewProj;


@vertex
fn vs_main(model: VertexInput, instance: InstanceInput ) -> VertexOutput {
	let model_matrix = mat4x4<f32> (
		instance.model_matrix_0,
		instance.model_matrix_1,
		instance.model_matrix_2,
		instance.model_matrix_3,
	);

	var out: VertexOutput;
	out.tex_coords = model.tex_coords;
	let world_position = model_matrix *vec4<f32>(model.position, 1.0);
	out.clip_position = view_proj.proj * view_proj.view * world_position;
	out.world_position = world_position.xyz;
	out.world_normal = (model_matrix * vec4<f32>(model.normal, 0.0)).xyz;
	return out;
}



@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let obj_color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
	let ambient_strength = 0.1;
	let ambient_light =  light.color * ambient_strength;
	
	let L = light.dir;
	let N = in.world_normal;
	let diffuse_light = max(dot(L, N), 0.0) * light.color;

	let result_color = (diffuse_light + ambient_light) * obj_color.xyz;

	return vec4<f32>(result_color, obj_color.a);
}
