
mod mesh;
mod gpu;
mod camera;
mod world;
mod texture;
mod planet;
mod utils;
mod math;
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
                    self.yaw_input = delta.x;
                    self.pitch_input = delta.y;
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
            _ => {}
        }
    }

    pub fn update_camera(&mut self, camera: &mut camera::Camera, delta_time: f32){
        camera.target = self.center;

        self.yaw += self.yaw_input * self.orbit_speed * delta_time;
        self.pitch += self.pitch_input * self.orbit_speed * delta_time;


        self.pitch = self.pitch.clamp(-PI/2.0, PI/2.0);

        // println!("yaw_input: {}, pitch_input: {}", self.yaw_input, self.pitch_input);
        // println!("yaw: {}, pitch: {}", self.yaw, self.pitch);

        //yaw pitch control camera rotation
        let quat =  Quaternion::rotation_y(self.yaw)* Quaternion::rotation_x(self.pitch) ;


        self.orbit_radius -= self.wheel_input * self.zoom_speed * delta_time;
        self.orbit_radius = self.orbit_radius.max(0.1);

        camera.eye = self.center + quat * Vec3::unit_z() * self.orbit_radius;

        //clear input
        self.yaw_input = 0.0;
        self.pitch_input = 0.0;
        self.wheel_input = 0.0;
    }
}


pub fn new_planet(config_file: std::fs::File, save_dir_path: path::PathBuf) -> Result<(), std::io::Error> {
    let planet_desc: planet::PlanetDescriptor = ron::de::from_reader(config_file).unwrap();
    //check if planet already exists
    let planet_dir_path = save_dir_path.join(&planet_desc.name);
    if planet_dir_path.exists() {
        return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "planet already exists"));
    }

    let _planet = planet::Planet::new(&planet_desc, save_dir_path).unwrap();
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

pub fn init_world(save_dir_path: path::PathBuf) -> Result<(), std::io::Error> {
    let world_desc = world::WorldDescriptor{
        name: utils::Name::new("new world"),
    };
    let _world = world::World::new(&world_desc, save_dir_path).unwrap();
    Ok(())
}


struct State{
    gpu_agent: gpu::GpuAgent,
    world: world::World,
    world_state: world::WorldState,
    camera: camera::Camera,
    camera_controller: CameraOrbitController,

    cube: mesh::Triangles,
    cube_state: mesh::TrianglesState,
    cube_render_pipeline: wgpu::RenderPipeline,
}

impl State {
    pub fn new(gpu_agent: gpu::GpuAgent, world: world::World) -> Self{
        let world_state = world::WorldState::new(&gpu_agent, &world);
        let camera = camera::Camera::new( [0.0, 0.0, 5.0].into(), [0.0, 0.0, 0.0].into(), Vec3::unit_y(), camera::Projection{
            aspect: gpu_agent.surface_aspect(),
            fovy: PI/4.0,
            znear: 0.1,
            zfar: 100.0,
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
        }
    }

    pub fn input(&mut self, event: &WindowEvent){
        self.camera_controller.input(event);
    }

    pub fn update(&mut self, delta_time: f32){
        self.camera_controller.update_camera(&mut self.camera, delta_time);
        self.camera.update();
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

            //render cube
            self.render_cube(&mut render_pass);
            
        }

        self.gpu_agent.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
  
    }

    fn render_cube<'a: 'b, 'b>(&'a self, render_pass: & mut wgpu::RenderPass<'b>){
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

pub async fn run(save_dir_path: path::PathBuf){
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let gpu_agent = gpu::GpuAgent::new(window).await;

    let world_desc = world::WorldDescriptor{
        name: Name::new("new_world"),
    };
    let world = world::World::new(&world_desc, std::path::PathBuf::from(save_dir_path)).unwrap();

   let mut state = State::new(gpu_agent, world);

    
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


