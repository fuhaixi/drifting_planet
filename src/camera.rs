
use vek::*;
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewProj{
    pub view_matrix: [[f32; 4]; 4],
    pub proj_matrix: [[f32; 4]; 4],
}

impl ViewProj{
    pub fn new(eye: Vec3<f32>, target: Vec3<f32>, up: Vec3<f32>, projection: Projection) -> Self{
        let view_matrix: [[f32; 4]; 4] = 
            Mat4::look_at_rh(eye, target, up).into_col_arrays();

        let proj_matrix: [[f32; 4]; 4] = projection.build_proj_matrix();
        
        Self{
            view_matrix,
            proj_matrix,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Projection{
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Projection{

    pub fn build_proj_matrix(&self) -> [[f32; 4];4] {
        ( Mat4::perspective_rh_zo(self.fovy, self.aspect, self.znear, self.zfar)).into_col_arrays()
    }
}

///camera is always update 
///so just cache uniform here
pub struct Camera{
    uniform: ViewProj,
    pub eye: Vec3<f32>,
    pub target: Vec3<f32>,
    pub up: Vec3<f32>,

    pub projection: Projection,
    
}

impl Camera{
    
    
    pub const PUSH_CONSTANT_RANGE: wgpu::PushConstantRange = wgpu::PushConstantRange{
        stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
        range: 0..std::mem::size_of::<ViewProj>() as u32,
    };

    pub fn new(eye: Vec3<f32>, target: Vec3<f32>, up: Vec3<f32>, projection: Projection) -> Self{
        Self{
            eye,
            target,
            up,
            
            projection,
            uniform: ViewProj::new(eye, target, up, projection)
        }
    }
    fn build_uniform(&self) -> ViewProj{
        
        ViewProj::new(self.eye, self.target, self.up, self.projection)
    }

    pub fn ndc_transform(&self, point: Vec3<f32>)-> Vec3<f32>{
        let view_matrix = Mat4::<f32>::from_col_arrays(self.uniform.view_matrix);
        let proj_matrix = Mat4::<f32>::from_col_arrays(self.uniform.proj_matrix);
        let p = (proj_matrix * view_matrix) * Vec4::from(point);
        p.xyz() / p.w
    }

    //return View dot Normal result and distance
    pub fn ray_dot(&self, position: Vec3<f32>, normal: Vec3<f32>) -> (f32, f32){
        let view_vector = self.eye - position;
        let distance = view_vector.magnitude();
        let d = (view_vector / distance).dot(normal);
        (d, distance)
    }

    pub fn is_point_visible(&self, point: Vec3<f32>) -> bool{
        let p = self.ndc_transform(point);
        
        return p.x< 1.0 && p.x > -1.0 && p.y < 1.0 && p.y > -1.0 && p.z >0.0 && p.z < 1.0;
    }

    pub fn get_uniform_data(&self) -> &[u8] {
        
        bytemuck::bytes_of(&self.uniform)
    }

    pub fn update(&mut self){
        self.uniform = self.build_uniform();
    }

}
