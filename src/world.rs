
use std::fs;
use crate::camera;
use crate::planet;
use crate::planet::PlanetDescriptor;
use crate::texture;
use crate::gpu;
use crate::mesh;

use vek::*;
use crate::utils::Name;

use serde::{Serialize, Deserialize};




pub struct World{
    pub name: Name,
    pub planets: Vec<PlanetDescriptor>,
    pub visible_planets: Vec<planet::Planet>,

    pub star_instances: Vec<mesh::TransformInstance>,
    pub star_triangles: mesh::Triangles,
}

#[derive(Serialize, Deserialize)]
pub struct WorldDescriptor{
    pub name: Name,
}



impl World{
    const STARS_SCATTER_RANGE: f32 = 50.0;
    const MAX_PLANETS: usize = 10;

    pub fn build(desc: &WorldDescriptor, world_dir_path: std::path::PathBuf) -> Result<(), std::io::Error>{
        if !(world_dir_path.exists() && world_dir_path.is_dir()){
            println!("Creating world directory: {:?}", world_dir_path);
            fs::create_dir_all(&world_dir_path)?;
        }

        //create world descriptor ron
        let world_desc_path = world_dir_path.join("world.ron");
        
        println!("Creating world descriptor: {:?}", world_desc_path);
        let mut world_desc_file = fs::File::create(&world_desc_path)?;
        
        ron::ser::to_writer_pretty(&mut world_desc_file, &desc, ron::ser::PrettyConfig::default()).unwrap();
    
        Ok(())
    }

    pub fn load_from_dir(world_dir_path: std::path::PathBuf) ->  Result<Self, std::io::Error> {
        //load desc
        let world_desc_path = world_dir_path.join("world.ron");
        let world_desc_file = fs::File::open(&world_desc_path).unwrap_or_else(|_| panic!("Failed to open world descriptor: {:?}", world_desc_path));
        let world_desc: WorldDescriptor = ron::de::from_reader(world_desc_file).unwrap();

        //generate stars
        use rand::Rng;
        use rand::distributions::Uniform;
        let mut rng = rand::thread_rng(); 
        let uniform = Uniform::new(-Self::STARS_SCATTER_RANGE, Self::STARS_SCATTER_RANGE);

        let star_instances: Vec<mesh::TransformInstance> = (0..300).map(|_i|{

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

        //iter world dir and load planets descriptors
        let mut planets = Vec::new();
        for entry in fs::read_dir(world_dir_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir(){
                let planet_desc_path = path.join("planet.ron");
                if planet_desc_path.exists(){
                    let planet_desc_file = fs::File::open(&planet_desc_path)?;
                    let planet_desc: PlanetDescriptor = ron::de::from_reader(planet_desc_file).unwrap();
                    planets.push(planet_desc);
                }
            }
        }

        let star_triangles = mesh::Triangles::create_polygon(0.2, 7);
        Ok(Self{
            name: world_desc.name.clone(),
            planets,
            star_instances,
            star_triangles,
            visible_planets: Vec::new(),
        })
    }

 
    

    #[allow(dead_code)]
    pub fn describe(&self) -> WorldDescriptor{
        WorldDescriptor{
            name: self.name.clone(),
        }
    }

    pub fn update(&mut self, delta_time: f32, camera: &camera::Camera, detail_cap: f32){
        for planet in self.visible_planets.iter_mut(){
            planet.update(delta_time, camera, detail_cap);
        }
    }

    pub fn focus_on_planet(&mut self, planet_name: &Name, world_dir_path: &std::path::PathBuf){
        //check if planet_name in planets
        if !self.planets.iter().any(|p| p.name == *planet_name){
            println!("Planet {} not found in world {}", planet_name, self.name);
            return;
        }

        println!("Focusing on planet {}", planet_name);
        //print descsriptor
        let planet_desc = self.planets.iter().find(|p| p.name == *planet_name).unwrap();
        println!("Planet descriptor: {:?}", planet_desc);

        let planet_path = world_dir_path.join(planet_name.as_str());
        let planet = planet::Planet::load_from_dir(planet_path, planet_name.clone()).unwrap();

        self.visible_planets = vec![planet];
    }


    
}



//hold gpu data of world
pub struct WorldState {
    stars_instance_buffer: wgpu::Buffer,
    stars_state: mesh::TrianglesState,
    stars_instances_count: u32,
    back_render_pipeline: wgpu::RenderPipeline,
    sun_render_pipeline: wgpu::RenderPipeline,

    visible_planets_state: Vec<planet::PlanetState>,
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


        let sun_shader = agent.create_shader_from_path("src/shaders/sun.wgsl");

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

        let world_back_shader = agent.create_shader_from_path( "src/shaders/world_back.wgsl");
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

            visible_planets_state: Vec::new(),
        }
    }

    pub fn update(&mut self, world: &World, agent: & gpu::GpuAgent) {
        // agent.queue.write_buffer(&self.stars_instance_buffer, 0, bytemuck::cast_slice(&world.star_instances));
        
    }

    pub fn set_visible_planets_state(&mut self, world: &World, agent: & gpu::GpuAgent){
        self.visible_planets_state.clear();
        for planet in world.visible_planets.iter(){
            self.visible_planets_state.push(planet::PlanetState::new(agent, planet));
        }
    }

    pub fn update_visible_planets_state(&mut self, world: &World, agent: & gpu::GpuAgent){
        for i in 0..self.visible_planets_state.len(){
            let planet_state = &mut self.visible_planets_state[i];
            let planet = &world.visible_planets[i];
            planet_state.update(agent, planet);
        }
    }

    

    pub fn render_back<'a>(&'a self, render_pass: & mut wgpu::RenderPass<'a>, camera: &camera::Camera) {
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

    pub fn render_sun<'a>(&'a self, render_pass: & mut wgpu::RenderPass<'a>, camera: &camera::Camera) {

        render_pass.set_pipeline(&self.sun_render_pipeline);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0, 
            camera.get_uniform_data(),
        );
        
        render_pass.draw(0..3, 0..1);
    }

    pub fn render_visible_planets<'a>(&'a self, render_pass: & mut wgpu::RenderPass<'a>, camera: &camera::Camera, world: &World, wireframe: bool){
        for i in 0..self.visible_planets_state.len(){
            let planet_state = &self.visible_planets_state[i];
            let planet = &world.visible_planets.get(i).unwrap_or_else(|| panic!("planet {} is visible but not loaded into world", i));
    
            planet_state.render_terrain(render_pass, camera, planet, wireframe);
         
        }
    }
}

//test random
#[cfg(test)]
mod test{
    use super::*;
    use rand::Rng;
    use rand::distributions::Uniform;

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