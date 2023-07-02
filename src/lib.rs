
mod mesh;
mod gpu;
mod camera;
mod world;
mod texture;
mod planet;
mod utils;
mod math;
mod user;
mod noise;
use std::path;
use utils::Name;
use vek::*;
use std::f32::consts::PI;
use std::fs;
use winit::{
    event::*,
    event_loop::{ EventLoop},
    window::{WindowBuilder},
};

///control camera with mouse draging orbiting around a center point
/// wheel to zoom in and out
pub struct CameraOrbitController{
    pub center: Vec3<f32>,
    pub orbit_radius: f32,
    pub quat: Quaternion<f32>,
    pub orbit_speed: f32,
    pub zoom_speed: f32,
    pub is_dragging: bool,

    pub yaw: f32,
    pub pitch: f32,
    pub wheel_input: f32,
    pub yaw_input: f32,
    pub pitch_input: f32,

    pub last_mouse_pos: Vec2<f32>,

    alt_offset_input: Vec2<f32>,
    alt_offset: Vec2<f32>,
    alt_hold: bool,
}

pub struct CameraFp{
    
}

impl CameraOrbitController{

    pub fn new(center: Vec3<f32>, orbit_radius: f32) -> Self{
        

        Self{
            center,
  
            orbit_speed: 1.0,
            zoom_speed: 50.0,
            is_dragging: false,
            yaw: 0.0,
            pitch: 0.0,
            orbit_radius,
            quat: Quaternion::identity(),
            last_mouse_pos: Vec2::zero(),
            wheel_input: 0.0,

            yaw_input: 0.0,
            pitch_input: 0.0,

            alt_hold: false,
            alt_offset_input: Vec2::zero(),
            alt_offset: Vec2::zero(),
        }
    }


    ///dragging the mouse will orbit camera around center point
    pub fn input(&mut self, event: &WindowEvent){
        match event{
            WindowEvent::MouseInput{state, button, ..} => {
                if *button == MouseButton::Left{
                    self.is_dragging = *state == ElementState::Pressed;
                }
            },
            WindowEvent::CursorMoved{position, ..} => {
                let mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                
                if self.is_dragging{
                    let delta = mouse_pos - self.last_mouse_pos;
                    if self.alt_hold{
                        self.alt_offset_input = delta;
                    }
                    else{

                        self.yaw_input = delta.x;
                        self.pitch_input = delta.y;
                    }
                }
                
                self.last_mouse_pos = mouse_pos;
            }

            //mouse wheel to zoom in and out
            WindowEvent::MouseWheel{delta, ..} => {
                match delta{
                    MouseScrollDelta::LineDelta(_, y) => {
                        self.wheel_input = *y ;
                    }
                    MouseScrollDelta::PixelDelta(pos) => {
                        self.wheel_input = pos.y as f32;
                    }
                }
            }


            //alt pressed to offset center point
            WindowEvent::KeyboardInput{input, ..} => {
                if let Some(key) = input.virtual_keycode{
                    match key{
                        VirtualKeyCode::LAlt => {
                            self.alt_hold = input.state == ElementState::Pressed;
                            if self.alt_hold == false{
                                self.alt_offset_input = Vec2::zero();
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    pub fn update_camera(&mut self, camera: &mut camera::Camera, delta_time: f32){
        camera.target = self.center;

        let center_offset;
        if self.alt_hold{
            self.alt_offset += self.alt_offset_input * delta_time * 0.5;
            let quat = Quaternion::rotation_3d(self.alt_offset.x, camera.get_right_axis()).rotate_3d(self.alt_offset.y, camera.get_up_axis());
            center_offset = (self.alt_offset.x * camera.get_right_axis() + self.alt_offset.y *camera.get_up_axis()) * self.orbit_radius * 0.1f32;
        }
        else{
            center_offset = Vec3::zero();
        }

        self.yaw += self.yaw_input * self.orbit_speed * delta_time;
        self.pitch += self.pitch_input * self.orbit_speed * delta_time;


        self.pitch = self.pitch.clamp(-PI/2.0 + 0.001, PI/2.0 - 0.001);

        // println!("yaw_input: {}, pitch_input: {}", self.yaw_input, self.pitch_input);
        // println!("yaw: {}, pitch: {}", self.yaw, self.pitch);

        //yaw pitch control camera rotation
        let quat =  Quaternion::rotation_y(self.yaw)* Quaternion::rotation_x(self.pitch) ;


        self.orbit_radius -= self.wheel_input * self.zoom_speed  * self.orbit_radius * delta_time * 0.1f32;
        self.orbit_radius = self.orbit_radius.max(0.1);
   
        camera.eye = self.center + quat * Vec3::unit_z() * self.orbit_radius + center_offset;
       

        //clear input
        self.yaw_input = 0.0;
        self.pitch_input = 0.0;
        self.wheel_input = 0.0;
        self.alt_offset_input = Vec2::zero();
    }
}


pub fn build_planet(config_file: std::fs::File, save_dir_path: path::PathBuf) -> Result<(), std::io::Error> {
    let planet_desc: planet::PlanetDescriptor = ron::de::from_reader(config_file).unwrap();
    let byte_need = planet_desc.calc_bytes_need();
    let mb = byte_need as f32 / 1024.0 / 1024.0;
    
    //check if planet already exists
    let planet_dir_path = save_dir_path.join(&planet_desc.name);
    if !planet_dir_path.exists() {
        fs::create_dir_all(&planet_dir_path).unwrap();
    }
    
    planet::Planet::build(&planet_desc, planet_dir_path).unwrap();
    
    println!("cost space: {} mb", mb);
    Ok(())
}

pub fn list_planets(save_path: path::PathBuf){
    let mut planet_names = Vec::new();
    for entry in fs::read_dir(save_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            planet_names.push(path.file_name().unwrap().to_str().unwrap().to_string());
        }
    }
    println!("Planets:");
    for name in planet_names{
        println!("{}", name);
    }
}

pub fn init_world(save_dir_path: path::PathBuf, name: String) -> Result<(), std::io::Error> {
    let world_desc = world::WorldDescriptor{
        name: utils::Name::new(&name),
    };
    let world_path = save_dir_path.join(&world_desc.name);
    world::World::build(&world_desc, world_path).unwrap();
    Ok(())
}


struct State{
    gpu_agent: gpu::GpuAgent,
    world: world::World,
    world_state: world::WorldState,
    camera: camera::Camera,
    camera_controller: CameraOrbitController,

    cube: mesh::Triangles,
    freeze_mode : bool,
    cube_state: mesh::TrianglesState,
    cube_render_pipeline: wgpu::RenderPipeline,
    world_dir_path: path::PathBuf,
    detail_cap: f32,
    fps_debug: bool,
    wireframe: bool,
}

impl State {
    pub fn new(gpu_agent: gpu::GpuAgent, world: world::World, world_dir_path: path::PathBuf) -> Self{
        let world_state = world::WorldState::new(&gpu_agent, &world);
        let camera = camera::Camera::new( [0.0, 0.0, 5.0].into(), [0.0, 0.0, 0.0].into(), Vec3::unit_y(), camera::Projection{
            aspect: gpu_agent.surface_aspect(),
            fovy: PI/4.0,
            znear: 0.1,
            zfar: 400.0,
        });
        let camera_controller = CameraOrbitController::new([0.0, 0.0, 0.0].into(), 10.0);

        let cube = mesh::Triangles::create_cube( 1.0);
        let cube_state = mesh::TrianglesState::new(&gpu_agent, &cube);

        let cube_pipeline_layout = gpu_agent.create_pipeline_layout(
            &[],
            &[camera::Camera::PUSH_CONSTANT_RANGE],
            "cube pipeline layout"
        );
        use mesh::VertexLayout;
        let cube_render_pipeline = gpu_agent.create_render_pipeline(
            &cube_pipeline_layout,
            &[mesh::Triangles::vertex_layout::<0>()],
            &gpu_agent.create_shader(include_str!("shaders/cube.wgsl"), "cube shader"),
            wgpu::PolygonMode::Line,
            gpu_agent.config.format,
            texture::Texture::DEPTH_FORMAT,
            Some(wgpu::Face::Back),
            "cube render pipeline"
        );

        


        Self{
            gpu_agent,
            world,
            world_state,
            camera,
            camera_controller,
            cube,
            cube_state,
            cube_render_pipeline,
            freeze_mode:false,
            world_dir_path,
            detail_cap: 1.0f32,
            fps_debug: false,
            wireframe: false,
        }
    }

    pub fn input(&mut self, event: &WindowEvent){
        self.camera_controller.input(event);

        //if press b print camera position
        if let WindowEvent::KeyboardInput { input, .. } = event {
            if let Some(key_code) = input.virtual_keycode {
                if input.state == ElementState::Pressed {
                    if key_code == VirtualKeyCode::B {
                        println!("camera position: {:?}", self.camera.eye);
                        //camera controller radius
                        println!("camera controller radius: {}", self.camera_controller.orbit_radius);
                    }
                    if key_code == VirtualKeyCode::F {
                        self.freeze_mode = !self.freeze_mode;
                    }
                }
            }

            //if press k print fps
            if let WindowEvent::KeyboardInput { input, .. } = event {
                if let Some(key_code) = input.virtual_keycode {
                    if input.state == ElementState::Pressed {
                        if key_code == VirtualKeyCode::K {
                            self.fps_debug = !self.fps_debug;
                        }
                    }
                }
            }

            //if press l toggle wireframe
            if let WindowEvent::KeyboardInput { input, .. } = event {
                if let Some(key_code) = input.virtual_keycode {
                    if input.state == ElementState::Pressed {
                        if key_code == VirtualKeyCode::L {
                            self.wireframe = !self.wireframe;
                        }
                    }
                }
            }
        }

    

        //prees +- control detail cap
        if let WindowEvent::KeyboardInput { input, .. } = event {
            if let Some(key_code) = input.virtual_keycode {
                if input.state == ElementState::Pressed {
                    if key_code == VirtualKeyCode::NumpadAdd {
                        self.detail_cap *= 2.0;
                        println!("detail cap: {}", self.detail_cap);
                    }
                    if key_code == VirtualKeyCode::NumpadSubtract {
                        self.detail_cap /= 2.0;
                        println!("detail cap: {}", self.detail_cap);
                    }
                }
            }
        }
    }

    pub fn update(&mut self, delta_time: f32){
        self.camera_controller.update_camera(&mut self.camera, delta_time);
        self.camera.update();

        if !self.freeze_mode{

            self.world.update(delta_time, &self.camera, self.detail_cap);
        }
        self.world_state.update_visible_planets_state(&self.world, &self.gpu_agent);

        if self.fps_debug{
            println!("fps: {}", 1.0/delta_time);
        }
    }

    pub fn focus(&mut self, planet_name: &Name){
        let planet_desc= self.world.planets.iter().find(|p| p.name == *planet_name).unwrap();
        self.camera_controller.center = planet_desc.position;
        self.camera_controller.orbit_radius = planet_desc.radius * 2.0;

        self.world.focus_on_planet(planet_name, &self.world_dir_path);
        self.world_state.set_visible_planets_state(&self.world, &self.gpu_agent);
    }

    pub fn render(&mut self, delta_time: f32) -> Result<(), wgpu::SurfaceError>{

        let output = self.gpu_agent.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.gpu_agent.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label: Some("Render Pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view:&view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 1.0,
                            }),
                            store: true,
                        },
                    })
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.gpu_agent.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });


            self.world_state.render_back(&mut render_pass, &self.camera);
            // self.world_state.render_sun(&mut render_pass, &self.camera);
            self.world_state.render_visible_planets( &mut render_pass, &self.camera, &self.world, self.wireframe);

            //render cube
            self.render_cube(&mut render_pass);
            
        }

        self.gpu_agent.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
  
    }

    fn render_cube<'a>(&'a self, render_pass: & mut wgpu::RenderPass<'a>){
        render_pass.set_pipeline(&self.cube_render_pipeline);
        render_pass.set_vertex_buffer(0, self.cube_state.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_state.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_push_constants(
            wgpu::ShaderStages::VERTEX_FRAGMENT,
            0,
            self.camera.get_uniform_data()
        );
        render_pass.draw_indexed(0..self.cube_state.index_count, 0, 0..1);
    }


    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>){
        self.gpu_agent.resize(new_size);

        self.camera.projection.aspect = self.gpu_agent.surface_aspect();
    }

    pub fn window(&self) -> &winit::window::Window{
        &self.gpu_agent.window
    }


}

pub async fn run_with_string(save_dir_path: path::PathBuf, focus_planet: Option<String>, world_name: &str,){
    let focus_planet = focus_planet.map(|s| Name::new(&s));
    run(save_dir_path, world_name, focus_planet).await;
}

pub async fn run(save_dir_path: path::PathBuf, world_name: &str, focus_planet: Option<Name>,){
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let gpu_agent = gpu::GpuAgent::new(window).await;

    let world_dir_path = save_dir_path.join(world_name);
    let world = world::World::load_from_dir( world_dir_path.clone()).unwrap();

    let mut state = State::new(gpu_agent, world, world_dir_path.clone());

    if let Some(planet_name) = focus_planet{
        state.focus(&planet_name);
    }

    
    let mut last_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| 
    {
        control_flow.set_poll();

        match event {
            Event::WindowEvent {
                event,
                ..
            } => {
                state.input(&event);
                match event {
                    WindowEvent::CloseRequested => control_flow.set_exit(),
                    
                    WindowEvent::Resized(physical_size) => {
                        state.resize(physical_size);
                    }
                    WindowEvent::ScaleFactorChanged {new_inner_size, .. } => {
                        state.resize(*new_inner_size);
                    }
                    _ => () 
                }
                
            }

            Event::RedrawRequested(window_id) if window_id == state.window().id() => {
                let now = std::time::Instant::now();
                let delta_time = now.duration_since(last_time).as_secs_f32();
                last_time = now;

                state.update(delta_time);
                match state.render(delta_time) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.gpu_agent.size),
                    Err(wgpu::SurfaceError::OutOfMemory) => control_flow.set_exit(),
                    Err(e) => eprintln!("{:?}", e)
                }
            }

            Event::MainEventsCleared => {
                state.window().request_redraw();
            }

            _ => {}
        }

    });
    
}

#[cfg(test)]
mod test{
    use  super::*;
    #[test]
    fn _run(){
        let save_dir_path = path::PathBuf::from("test/save");
        if !save_dir_path.exists(){
            fs::create_dir_all(&save_dir_path).unwrap();
        }

        pollster::block_on(run(save_dir_path, "world_A", Some(Name::new("AA"))));

    }

    
    fn _build_world(level: u8, grid_segment_num: u32) -> f32{
        let save_dir_path = path::PathBuf::from("test/save");
        let world_dir_path =save_dir_path.join("world_A");
        if !world_dir_path.exists(){
            fs::create_dir_all(&world_dir_path).unwrap();
        }

        let world_desc = world::WorldDescriptor{
            name: Name::new("world_A"),
        };

        let planet_desc = planet::PlanetDescriptor{
            name: utils::Name::new("AA"),
            radius: 50.0,
            lod_level: level,
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
            mesh_grid_segment_num: grid_segment_num,
            terrain_noise: utils::NoiseDescriptor{
                seed: 0,
                frequency: 10.0,
                lacunarity: 2.0,
                persistence: 0.5,
                octaves: 8,
            },
            elevation_scale: 3.0,
        };

  

        let planet_dir = world_dir_path.join("AA");
        if !planet_dir.exists(){
            fs::create_dir_all(&planet_dir).unwrap();
        }

        planet::Planet::build(&planet_desc, planet_dir).unwrap();


        world::World::build(&world_desc, world_dir_path).unwrap();
        let byte_need = planet_desc.calc_bytes_need();
        let mb = byte_need as f32 / 1024.0 / 1024.0;
        return  mb ;
    }

    #[test]
    fn _build_world_0(){
        //set level = 6 and change grid_segment_num from 4 to 16
        //record time and memory usage
        let mut infos = Vec::new();

        for grid_segment_num in 4..=16{
            //time
            let start = std::time::Instant::now();
            let mb = _build_world(5, grid_segment_num);
            let end = std::time::Instant::now();
            let duration = end.duration_since(start);
            let duration = duration.as_secs_f32();
            infos.push((grid_segment_num, duration, mb));
        }
        
        for (grid_segment_num, duration, mb) in infos{
            println!("level 5 ,grid_segment_num {} => duration: {}, mb: {}", grid_segment_num, duration, mb);
        }
    }

    #[test]
    fn _build_world_1(){
        //set level = 6 and change grid_segment_num from 4 to 16
        //record time and memory usage
        let mut infos = Vec::new();

        for level in 4..=8{
            //time
            let start = std::time::Instant::now();
            let mb = _build_world(level, 8);
            let end = std::time::Instant::now();
            let duration = end.duration_since(start);
            let duration = duration.as_secs_f32();
            infos.push((level, duration, mb));
        }
        
        for (level, duration, mb) in infos{
            println!("level {} ,grid_segment_num 8 => duration: {}, mb: {}", level, duration, mb);
        }
    }

    #[test]
    fn _build_world_2(){
        let mb = _build_world(4, 32);
        println!("level 5 ,grid_segment_num 64 => mb: {}", mb);
    }
}



