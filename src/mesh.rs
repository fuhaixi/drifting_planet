use vek::*;
use std::f32::consts::PI;
use serde::{Serialize, Deserialize};
use wgpu::util::DeviceExt;
use crate::math::AxisNormal;
use crate::gpu;


pub const QUAD_CORNERS:[(f32, f32); 4] = [(-1.0, 1.0), (1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)];

pub trait VertexLayout{
    fn vertex_layout<const LOACTION: u32>() -> wgpu::VertexBufferLayout<'static>;
}

pub trait Vertex {
    
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct MeshVertex{
    pub pos: [f32; 3],
    pub tex_coord: [f32; 2],
    pub normal: [f32; 3]
}



impl Default for MeshVertex{
    fn default() -> Self{
        Self{
            pos: [0.0, 0.0, 0.0],
            tex_coord: [0.0, 0.0],
            normal: [0.0, 0.0, 0.0]
        }
    }
}

impl VertexLayout for MeshVertex {

    ///location length : 3
    fn vertex_layout<const LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout{
            array_stride: mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute{
                    offset: 0,
                    shader_location: 0 + LOCATION,
                    format: wgpu::VertexFormat::Float32x3
                },
                wgpu::VertexAttribute{
                    offset: mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1 + LOCATION,
                    format: wgpu::VertexFormat::Float32x2
                },
                wgpu::VertexAttribute{
                    offset: mem::size_of::<[f32; 5]>() as u64,
                    shader_location: 2 + LOCATION,
                    format: wgpu::VertexFormat::Float32x3
                }
            ]
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Mesh{
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,

}

pub struct MeshState{
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl MeshState {
    pub fn new(agent: &gpu::GpuAgent, mesh: &Mesh) -> Self {
        let vertex_buffer = agent.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = agent.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self{
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        }
    }


}



//Tc: translation and color
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct TcInstance{
    pub translation: [f32; 4],
    pub rgba: [f32; 4]
}

impl VertexLayout for TcInstance{


    ///location length: 2
    fn vertex_layout<const LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<TcInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute{
                    offset: 0,
                    shader_location: 0 + LOCATION,   
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute{
                    offset: std::mem::size_of::<[f32; 4]>() as u64,
                    shader_location: 1 + LOCATION ,
                    format: wgpu::VertexFormat::Float32x4
                }
            ],

        }

    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct  TransformInstance{
    pub mat4: [[f32; 4]; 4]
}

impl VertexLayout for TransformInstance {
    ///location length: 1
    fn vertex_layout<const LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<TransformInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute{
                    offset: 0,
                    shader_location: 0 + LOCATION,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute{
                    offset: std::mem::size_of::<[f32; 4]>() as u64,
                    shader_location: 1 + LOCATION,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute{
                    offset: std::mem::size_of::<[[f32; 4]; 2]>() as u64,
                    shader_location: 2 + LOCATION,
                    format: wgpu::VertexFormat::Float32x4
                },
                wgpu::VertexAttribute{
                    offset: std::mem::size_of::<[[f32; 4]; 3]>() as u64,
                    shader_location: 3 + LOCATION,
                    format: wgpu::VertexFormat::Float32x4
                }
            ]
        }
    }
}

impl TcInstance{
    pub fn from_position_rgb(position: [f32; 3], rgb: [f32; 3]) -> Self{
        Self{
            translation: [position[0], position[1], position[2], 1.0],
            rgba: [rgb[0], rgb[1], rgb[2], 1.0]
        }
    }
}

//position and indices

#[derive(Serialize, Deserialize)]
pub struct Triangles(pub Vec<[f32; 3]>, pub Vec<u32>);

impl VertexLayout for Triangles{
    ///location length: 1
    fn vertex_layout<const LOCATION: u32>() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;

        wgpu::VertexBufferLayout{
            array_stride: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute{
                    offset: 0,
                    shader_location: 0 + LOCATION,
                    format: wgpu::VertexFormat::Float32x3
                }
            ]
        }
    }
}

pub struct TrianglesState{
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl TrianglesState{
    pub fn new(agent: &gpu::GpuAgent, triangles: &Triangles) -> Self{
        let vertex_buffer = agent.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&triangles.0),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = agent.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&triangles.1),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self{
            vertex_buffer,
            index_buffer,
            index_count: triangles.1.len() as u32,
        }
    }
}


impl Mesh{
   
    pub fn create_axis_normal_terrain_plane(
            axis: AxisNormal,
            start: Vec2<f32>,
            size: f32,
            offset: f32,
            subdivision: u32,
        ) -> Self {
        
        let triangles = Triangles::create_axis_normal_terrain_plane(axis, start, size, offset, subdivision);
        let rows:u32 = subdivision*2 +1; 
        let len :u32 = triangles.0.len() as u32;
  
        let vertices: Vec<MeshVertex> = (0..len).into_iter().map(|i|{
            MeshVertex{
                pos: triangles.0[i as usize],
                normal: axis.normal(),
                tex_coord: [(i % rows) as f32 / rows as f32, i as f32 / rows as f32],
            }
        }).collect();


        Self{
            vertices,
            indices: triangles.1,
            
        }

        
    }

    pub fn from_triangles(triangles: Triangles) -> Self{
        let len :u32 = triangles.0.len() as u32;
  
        let vertices: Vec<MeshVertex> = (0..len).into_iter().map(|i|{
            MeshVertex{
                pos: triangles.0[i as usize],
                normal: [0.0, 0.0, 0.0],
                tex_coord: [0.0, 0.0],
            }
        }).collect();

        Self{
            vertices,
            indices: triangles.1,
        }
    }
}


impl Triangles{
    
    fn create_axis_normal_terrain_plane(
        axis: AxisNormal, 
        start:Vec2<f32>,
        size :f32,
        offset: f32,
        subdivision: u32
    ) -> Self{
        let tangent = Vec3::from( axis.tangent());
        let btangent = Vec3::from( axis.btangent());
        let normal = Vec3::from( axis.normal());
        let subdivision = subdivision * 2;
        
        let start_point = tangent * start.x + btangent * start.y + normal*offset ;
        let points_num = (1 + 2 * subdivision).pow(2); 
        let cell_size:f32 = size / (subdivision as f32);
        let mut pos_arr: Vec<[f32; 3]> = vec![[0.0; 3]; points_num as usize];

        let subs = subdivision * 2 +1;
        for xi in 0..subs{
            for yi in 0..subs{//loop each vertex
                let point = start_point + (tangent * xi as f32 + btangent * yi as f32)* cell_size;
                pos_arr[(yi * subs + xi) as usize] = point.into_array();
            }
        }

        let mut indices = Vec::<u32>::new();
        for xi in 0..subdivision{
            for yi in 0..subdivision{//loop each quad
                let a = xi + yi * subs;
                let b = a + 1;
                let c = a + subs;
                let d = b + subs;
                let quad_type = (xi % 2) ^ (yi % 2);
                if quad_type == 0 {
                    indices.append(&mut vec![a, c, d, a, d, b]);
                }
                else{
                    indices.append(&mut vec![a, c, b, c, d, b]);
                }
            }
        }
        
        Self(pos_arr, indices)
    }

    pub fn create_grid_on_unit_cube(index_x: u32, index_y: u32, axis: AxisNormal, segment_num: u32, grid_segment_num: u32) -> Self{

        let point_num = 1 + grid_segment_num;

        let size = 2.0 / (segment_num  as f32);
        let grid_cell_size = size / (grid_segment_num as f32);
        
        let mut pos_arr: Vec<[f32; 3]> = vec![[0.0; 3]; point_num.pow(2) as usize];
        let start_point =  Vec3::<f32>::new(index_x as f32 * size - 1.0, index_y as f32 * size -1.0, 1.0);
        let mat3 =  Mat3::from_col_arrays(axis.mat3_col_arrays());
        for yi in 0..point_num{
            for xi in 0..point_num{//loop each vertex
                let dx = xi as f32 * grid_cell_size;
                let dy = yi as f32 * grid_cell_size;
                let point: Vec3<f32> = start_point + [dx, dy, 0.0];
                pos_arr[(yi * point_num + xi) as usize] = (mat3 * point).into_array();
                
            }
        }
        
        let mut indices:Vec<u32> = Vec::new();
        for yi in 0..grid_segment_num{
            for xi in 0..grid_segment_num{//loop each four quad
                let a = xi + yi * point_num;
                let b = a + 1;
                let c = a + point_num;
                let d = b + point_num;
                let quad_type = (xi % 2) ^ (yi % 2);
                if quad_type == 0 {
                    indices.append(&mut vec![a, d, c, a, b, d]);
                }
                else{
                    indices.append(&mut vec![a, b, c, c, b, d]);
                }
                
            }
        }

        Self(pos_arr, indices)
    }

    // order z axis
    pub fn create_polygon(radius: f32, edge_num: u32) -> Self{

        let center = Vec3::zero();
        let pointer = Vec3::<f32>::unit_y() * radius;
        let mut positions: Vec<[f32; 3]> = vec![center.into_array(), pointer.into_array()];
        let mut indices: Vec<u32> = Vec::new();
        for i in 0..edge_num - 1 {
            let rad = (i+1) as f32 * 2.0 * PI / (edge_num as f32);
            let quat = Quaternion::<f32>::rotation_3d(rad, -Vec3::<f32>::unit_z());
            positions.push((quat * pointer).into_array());
            indices.append(&mut vec![0, i+1, i+2]);
        }
        indices.append(&mut vec![0, edge_num, 1]);
        
        Self(positions, indices)
    }

    #[allow(dead_code)]
    pub fn create_cube(size: f32) -> Self{
        let half_size = size / 2.0;
        let positions = vec![
            [-half_size, -half_size, -half_size],
            [-half_size, -half_size, half_size],
            [-half_size, half_size, -half_size],
            [-half_size, half_size, half_size],
            [half_size, -half_size, -half_size],
            [half_size, -half_size, half_size],
            [half_size, half_size, -half_size],
            [half_size, half_size, half_size],
        ];
        let indices = vec![
            0, 1, 2, 1, 3, 2,
            4, 6, 5, 5, 6, 7,
            0, 2, 4, 4, 2, 6,
            1, 5, 3, 5, 7, 3,
            0, 4, 1, 4, 5, 1,
            2, 3, 6, 3, 7, 6,
        ];
        Self(positions, indices)
    }

    #[allow(dead_code)]
    pub fn save_obj_file(&self, path: &str){
        let mut file = std::fs::File::create(path).unwrap();
        use std::io::Write;
        for pos in &self.0{
            file.write_all(format!("v {} {} {}\n", pos[0], pos[1], pos[2]).as_bytes()).unwrap();
        }
        for i in (0..self.1.len()).step_by(3){
            file.write_all(format!("f {} {} {}\n", self.1[i]+1, self.1[i+1]+1, self.1[i+2]+1).as_bytes()).unwrap();
        }
    }

    pub fn merged(&self, other: &Self) -> Self{
        let mut pos_arr = self.0.clone();
        let mut indices = self.1.clone();
        let offset = pos_arr.len() as u32;
        pos_arr.append(&mut other.0.clone());
        indices.append(&mut other.1.iter().map(|i| i + offset).collect());
        Self(pos_arr, indices)
    }

    pub fn merge(&mut self, other: &Self){
        let offset = self.0.len() as u32;
        self.0.append(&mut other.0.clone());
        self.1.append(&mut other.1.iter().map(|i| i + offset).collect());
    }

    
}



//test save obj file

#[cfg(test)]
mod tests {
    use crate::math;

    use super::*;
    #[test]
    fn test_save_obj_file(){
        let triangles = Triangles::create_polygon(1.0, 7);
        triangles.save_obj_file("polygon.obj");

        let terrain = Triangles::create_axis_normal_terrain_plane(AxisNormal::Y, Vec2::zero()-1.0, 2.0, 0.0, 2);
        terrain.save_obj_file("terrain.obj");

  
    }


    #[test]
    fn test_cobe_wrap(){
        let mut terrain = Triangles::create_axis_normal_terrain_plane(AxisNormal::Y, Vec2::zero()-1.0, 2.0, 0.0, 2);
        for pos in terrain.0.iter_mut(){

            let x = Vec2::new(pos[0], pos[2]);
            
            
            
        }
    }

    #[test]
    fn test_planet_gen(){
        let planet = gen_planet(4, 10, true);
        planet.save_obj_file("test/test_objs/planet.obj");
        let planet_no_wrap = gen_planet(4, 10, false);
        planet_no_wrap.save_obj_file("test/test_objs/planet_no_wrap.obj");
        
    }

    fn gen_planet(segs: u32, grid_segs: u32, use_wrap: bool) -> Triangles{
        let mut cube_grid = Triangles(vec![], vec![]);
        //merge six face grid into cube grid
        for i in 0..6{
            let axis = AxisNormal::from_u32(i);

            
            for xi in 0..segs{
                for yi in 0..segs{
                    let mut face_grid = Triangles::create_grid_on_unit_cube(xi, yi , axis, segs, grid_segs);

                    //cobe wrap
                    for i in 0..face_grid.0.len(){
                        if use_wrap{
                            math::cobe_wrap_with_axis(&mut face_grid.0[i], axis);
                        }
                       
                       //normalize positions
                       face_grid.0[i] = Vec3::<f32>::from(face_grid.0[i]).normalized().into_array();
                    }

                    cube_grid.merge(&face_grid);
                }
            }
            
        }

        cube_grid
    }

}

