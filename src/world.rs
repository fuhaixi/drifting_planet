
use std::fs;
use crate::camera;
use crate::planet;
use crate::texture;
use crate::gpu;
use crate::mesh;

use vek::*;
use crate::utils::Name;

use serde::{Serialize, Deserialize};

pub struct WorldAgent{
    
}

impl WorldAgent{
    pub fn load_from_dir(world_dir: std::path::PathBuf){
        
    }

    pub fn init_wolrd(dir: std::path::PathBuf, world_name: Name, seed: u32){
        
    }

    
}

pub struct World{
    pub dir_path: std::path::PathBuf,
    pub name: Name,
    pub planets: Vec<planet::Planet>,
    pub star_instances: Vec<mesh::TransformInstance>,
    pub star_triangles: mesh::Triangles,
}

#[derive(Serialize, Deserialize)]
pub struct WorldDescriptor{
    pub name: Name,
}

struct WorldInfo{
    pub planets: Vec<Name>,
    pub sun_color: [f32; 3],
}

impl World{
    const STARS_SCATTER_RANGE: f32 = 50.0;
    const MAX_PLANETS: usize = 10;

    pub fn new(desc: &WorldDescriptor, save_path: std::path::PathBuf) -> Result<Self, std::io::Error>{
        let dir_path = save_path.join(desc.name.as_str());
        
        if !(dir_path.exists() && dir_path.is_dir()){
            println!("Creating world directory: {:?}", dir_path);
            fs::create_dir_all(&dir_path)?;
        }
       
        
        //generate stars
        use rand::Rng;
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng(); 
        let uniform = Uniform::new(-Self::STARS_SCATTER_RANGE, Self::STARS_SCATTER_RANGE);

        let star_instances: Vec<mesh::TransformInstance> = (0..300).map(|i|{

            let random_rotation = Quaternion::<f32>::rotation_from_to_3d(Vec3::unit_z(), Vec3::new(
                rng.sample(uniform),
                rng.sample(uniform),
                rng.sample(uniform),
            ));
            let position = random_rotation *Vec3::unit_z() * 80.0;
            //create transform point at zero
            let transform = Mat4::from( Transform{
                position,
                orientation: Quaternion::rotation_from_to_3d(Vec3::unit_z(), position),
                scale: Vec3::one(),
            }).into_col_arrays();
            
            
            mesh::TransformInstance{
                mat4: transform
            }
        }).collect();

        let star_triangles = mesh::Triangles::create_polygon(0.2, 7);
        Ok(Self{
            name: desc.name.clone(),
            planets: Vec::new(),
            star_instances,
            star_triangles,
            dir_path,
        })

    }
    

    pub fn describe(&self) -> WorldDescriptor{
        WorldDescriptor{
            name: self.name.clone(),
        }
    }

    pub fn update(&mut self, delta_time: f32, camera: &camera::Camera){
        for planet in self.planets.iter_mut(){
            planet.update(delta_time, camera);
        }
    }

    /// save to the path
    pub fn save(&self) {
        //save description to ron file
        let desc = self.describe();
        let ron_path = self.dir_path.join("world.ron");
        
        ron::ser::to_writer(
            fs::File::create(ron_path).unwrap(),
            &desc
        ).unwrap();
        
        for planet in self.planets.iter() {
            planet.save();
        }
        
    }
    
}



//hold gpu data of world
pub struct WorldState {
    stars_instance_buffer: wgpu::Buffer,
    stars_state: mesh::TrianglesState,
    stars_instances_count: u32,
    back_render_pipeline: wgpu::RenderPipeline,
    sun_render_pipeline: wgpu::RenderPipeline,
}



impl WorldState{
    pub fn new(agent: & gpu::GpuAgent, world: & World) -> Self{
        
        let stars_instance_buffer = agent.create_buffer(&world.star_instances, wgpu::BufferUsages::VERTEX, "stars instance buffer");
        let stars_state = mesh::TrianglesState::new(agent, &world.star_triangles);

        let stars_instances_count = world.star_instances.len() as u32;
        
        let sun_render_pipeline_layout = agent.create_pipeline_layout(
            &[],
            &[camera::Camera::PUSH_CONSTANT_RANGE],
            "sun render pipeline layout"
        );


        let sun_shader = agent.create_shader_from_path("shaders/sun.wgsl");

        let sun_render_pipeline = agent.create_render_pipeline(
            &sun_render_pipeline_layout,
            &[],
            &sun_shader,
            wgpu::PolygonMode::Fill,
            agent.config.format,
            texture::Texture::DEPTH_FORMAT,
            Some(wgpu::Face::Back),
            "sun render pipeline",
        );

        
        let back_render_pipeline_layout =  agent.create_pipeline_layout(
            &[],
            &[camera::Camera::PUSH_CONSTANT_RANGE],
            "back render pipeline",
        );
        
        use mesh::VertexLayout;

        let world_back_shader = agent.create_shader_from_path( "shaders/world_back.wgsl");
        let back_render_pipeline = agent.create_render_pipeline(
            &back_render_pipeline_layout,
            &[mesh::Triangles::vertex_layout::<0>(), mesh::TransformInstance::vertex_layout::<1>()],
            &world_back_shader,
            wgpu::PolygonMode::Fill,
            agent.config.format,
            texture::Texture::DEPTH_FORMAT,
            None,
            "world back render pipline"
        );
        
                
        Self{
            stars_instance_buffer,
            stars_state,
            stars_instances_count,
            sun_render_pipeline,
            back_render_pipeline,
        }
    }

    pub fn update(&mut self, world: &World, agent: & gpu::GpuAgent) {
        agent.queue.write_buffer(&self.stars_instance_buffer, 0, bytemuck::cast_slice(&world.star_instances));

    }

    

    pub fn render_back<'a: 'b, 'b>(&'a self, render_pass: & mut wgpu::RenderPass<'b>, camera: &camera::Camera) {
        render_pass.set_pipeline(&self.back_render_pipeline);
        render_pass.set_vertex_buffer(0, self.stars_state.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.stars_instance_buffer.slice(..));
        render_pass.set_index_buffer(self.stars_state.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0, 
            camera.get_uniform_data(),
        );
        render_pass.draw_indexed(0..self.stars_state.index_count, 0, 0..self.stars_instances_count);
    }

    pub fn render_sun<'a: 'b, 'b>(&'a self, render_pass: & mut wgpu::RenderPass<'b>, camera: &camera::Camera) {

        render_pass.set_pipeline(&self.sun_render_pipeline);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0, 
            camera.get_uniform_data(),
        );
        
        render_pass.draw(0..3, 0..1);
    }
}

//test random
mod test{
    use super::*;
    use rand::Rng;
    use rand::distributions::Uniform;
    use std::path::PathBuf;

    #[test]
    fn test_world(){
        let mut rng = rand::thread_rng();
        
        //gennerate random position
        let pos_range = Uniform::new(-100.0, 100.0);
        let pos_x = rng.sample(pos_range);
        let pos_y = rng.sample(pos_range);
        let pos_z = rng.sample(pos_range);

        let pos = Vec3::<f32>::new(pos_x, pos_y, pos_z);

        println!("pos: {:?}", pos);

    }
}