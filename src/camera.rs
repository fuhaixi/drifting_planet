
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

///camera controller
/// mouse move to rotate camera
pub struct CameraFreeController{
    pub camera_base_state: Camera,
    pub speed: f32,
    pub sensitivity: f32,
    yaw: f32,
    pitch: f32,
}

impl CameraFreeController{
    pub fn new(camera: Camera, speed: f32, sensitivity: f32) -> Self{
        Self{
            camera_base_state: camera,
            speed,
            sensitivity,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera){

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
    
    pub fn get_up_axis(&self) -> Vec3<f32> {
        let view_matrix = self.uniform.view_matrix;
        Vec3::new(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1])
    }

    pub fn get_forward_axis(&self) -> Vec3<f32> {
        let view_matrix = self.uniform.view_matrix;
        Vec3::new(view_matrix[0][2], view_matrix[1][2], view_matrix[2][2])
    }

    pub fn get_right_axis(&self) -> Vec3<f32> {
        let view_matrix = self.uniform.view_matrix;
        Vec3::new(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0])
    }
    
    pub const PUSH_CONSTANT_RANGE: wgpu::PushConstantRange = wgpu::PushConstantRange{
        stages: wgpu::ShaderStages::VERTEX_FRAGMENT,
        range: 0..std::mem::size_of::<ViewProj>() as u32,
    };

    pub fn size_of_push_constant() -> u32{
        std::mem::size_of::<ViewProj>() as u32
    }

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

    pub fn calc_depth(&self, point: Vec3<f32>) -> f32{
        let p = self.ndc_transform(point);
        p.z
    }

}

///
pub struct Cam{
    pub basis: Mat3<f32>,
    pub position: Vec3<f32>,
    pub projection: Projection,
}

impl Cam{
    pub fn build_viewproj(&self) -> ViewProj{
        let basis = self.basis.into_col_arrays();
        let view_matrix = Mat4::from_col_array([
            basis[0][0], basis[1][0], basis[2][0], 0.0,
            basis[0][1], basis[1][1], basis[2][1], 0.0,
            basis[0][2], basis[1][2], basis[2][2], 0.0,
            -self.position.dot(Vec3::new(basis[0][0], basis[1][0], basis[2][0])),
            -self.position.dot(Vec3::new(basis[0][1], basis[1][1], basis[2][1])),
            -self.position.dot(Vec3::new(basis[0][2], basis[1][2], basis[2][2])),
            1.0,
        ]);

        let proj_matrix = Mat4::perspective_rh_zo(self.projection.fovy, self.projection.aspect, self.projection.znear, self.projection.zfar);
        ViewProj{
            view_matrix: view_matrix.into_col_arrays(),
            proj_matrix: proj_matrix.into_col_arrays(),
        }
    }


}

mod test{
    

    #[test]
    fn test_ndc_transform(){
        use super::*;
        let camera = Camera::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Projection{
                aspect: 1.0,
                fovy: std::f32::consts::FRAC_PI_2,
                znear: 0.1,
                zfar: 100.0,
            }
        );
        let p = camera.ndc_transform(Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(p, Vec3::new(0.0, 0.0, 0.0));
    }
}
