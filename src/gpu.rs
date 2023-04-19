use wgpu::util::DeviceExt;
use crate::texture;



pub struct GpuAgent{
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,
    pub window: winit::window::Window,
    pub depth_texture: texture::Texture,

    //fst: full screen triangle
    pub fst_buffer: wgpu::Buffer,
    pub fst_vertex_layout: wgpu::VertexBufferLayout<'static>,
}

impl GpuAgent{

    pub fn render_full_screen_triangle<'a: 'b, 'b>(&'a self, render_pass: &'b mut wgpu::RenderPass<'b>){
        render_pass.set_vertex_buffer(0, self.fst_buffer.slice(..));
        render_pass.draw(0..3, 0..1);
    }
    
    pub async fn new(window: winit::window::Window) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor{
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default()
        });
    
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let mut limits = wgpu::Limits::default();
        limits.max_push_constant_size = 256;
        
    
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor{
            features: wgpu::Features::POLYGON_MODE_LINE | wgpu::Features::PUSH_CONSTANTS,
            limits,
            label: None,
        }, None).await.unwrap();
        
        let surface_caps = surface.get_capabilities(&adapter);
    
        let surface_format = surface_caps.formats.iter()
            .copied()
            .filter(|f| f.describe().srgb)
            .next()
            .unwrap_or(surface_caps.formats[0]);
    
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        
        surface.configure(&device, &config);
    
        let depth_texture = texture::Texture::create_depth_texture(&device, &config, "depth_texture");
        
        let fst: [f32; 9] = 
            [-1.0, -1.0, 0.0, 2.0, -1.0, 0.0, -1.0, 2.0, 0.0];
        let fst_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("full screen triangele buffer"),
            contents: bytemuck::cast_slice(&fst),
            usage: wgpu::BufferUsages::VERTEX,
        });
    
        let fst_vertex_layout = wgpu::VertexBufferLayout{
            array_stride: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3
                }
            ]
        };
    
    
    
        Self{
            window,
            size,
            config, 
            surface,
            device,
            queue,
            depth_texture,
            fst_buffer,
            fst_vertex_layout,
        }
    }

    pub fn surface_aspect(&self) -> f32{
        self.config.width as f32 / self.config.height as f32
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>){
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture = texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    pub fn create_shader(&self, shader_source: &str, label: &str) -> wgpu::ShaderModule{
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        })
    }

    pub fn create_shader_from_path(&self, path: &str) -> wgpu::ShaderModule{
        let shader_source = std::fs::read_to_string(path).unwrap();
        self.create_shader(&shader_source, path)
    }


    pub fn create_buffer<A: bytemuck::NoUninit>(&self, data: &[A], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    } 

    pub fn create_pipeline_layout(&self,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        push_constant_ranges: &[wgpu::PushConstantRange], label: &str
    ) -> wgpu::PipelineLayout {

        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            bind_group_layouts,
            push_constant_ranges,
            label: Some(label),
        })
    }   

    pub fn create_render_pipeline(
        &self,
        layout: &wgpu::PipelineLayout,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader_module: &wgpu::ShaderModule,
        polygon_mode: wgpu::PolygonMode,
        color_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
        cull_mode: Option<wgpu::Face>,
        label: &str,
    ) -> wgpu::RenderPipeline{


        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState{
                module: &shader_module,
                entry_point: "vs_main",
                buffers: vertex_layouts,
            },
            primitive: wgpu::PrimitiveState{
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                polygon_mode,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState{
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                bias: wgpu::DepthBiasState::default(),
                stencil: wgpu::StencilState::default(),
            }),
            multisample: wgpu::MultisampleState{
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState{
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState{
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::all(),
                })],
            }),
            multiview: None,
        })
    }


}




