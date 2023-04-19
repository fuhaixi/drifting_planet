struct ViewProj {
	view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct VertexInput {
	@location(0) position: vec3<f32>,
}

struct TransformInstanceInput{
    @location(1) model_matrix_0: vec4<f32>,
	@location(2) model_matrix_1: vec4<f32>,
	@location(3) model_matrix_2: vec4<f32>,
	@location(4) model_matrix_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

var<push_constant> view_proj: ViewProj;


@vertex
fn vs_main(model: VertexInput, instance: TransformInstanceInput ) -> VertexOutput {

    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0, 
        instance.model_matrix_1, 
        instance.model_matrix_2, 
        instance.model_matrix_3,
    );

	let world_position =  model_matrix * vec4<f32>( model.position.xyz, 1.0);
    let view_position  = view_proj.view * vec4<f32>(world_position.xyz, 0.0);

	var out: VertexOutput;  
    out.clip_position = view_proj.proj * vec4(view_position.xyz, 1.0);
    out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
	return out;
}


@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
