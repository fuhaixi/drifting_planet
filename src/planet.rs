
use std::cell::RefCell;
use std::fs;
use crate::camera;
use crate::math;
use crate::texture;
use crate::utils;
use serde::{Serialize, Deserialize};
use crate::gpu;
use crate::mesh;
use std::io::prelude::*;
use lru;
use vek::*;
use crate::utils::Name;
struct ChunkNode{
    pub terrain_chunk_index: u32,
    pub children: Option<[Box<ChunkNode>; 4]>,
    
}



pub struct TerrainChunkHead{
    pub raw_corners: [[f32; 2];4],
    pub axis_normal: math::AxisNormal,
    pub corners: [[f32; 3]; 4],
    pub index: u32,
    pub depth: u8,
    pub detail_value: f32,
    pub subdivision: u32,

}

impl TerrainChunkHead{
    pub fn calc_detail(&self, camera: &camera::Camera, center: Vec3<f32>) -> f32{
        self.corners.map(|pos|{
            let position = Vec3::from(pos);
            let normal: Vec3<f32> = position - center;
            let (dotre, distance) = camera.ray_dot(position, normal);
            let detail_value = if dotre < 0.0{
                0.0f32
            } else{
                
                dotre * 1.0 / (1.0 + distance)
            };
            detail_value
        }).iter().fold(f32::MIN, |max, x|max.max(*x))
    }

  
    
}


pub struct TerrainChunk{
    mesh: mesh::Mesh,
}

impl TerrainChunk{
    pub fn to_writer<W>(&self, writer: &mut W) -> Result<(), std::io::Error> where W: std::io::Write{
        writer.write_all(bytemuck::cast_slice(&self.mesh.vertices))?;
        writer.write_all(bytemuck::cast_slice(&self.mesh.indices))?;
        Ok(())
    }

    pub fn from_reader<R>(reader: &mut R) -> Result<Self, std::io::Error> where R: std::io::Read{
        let mut vertices = vec![mesh::MeshVertex::default(); CHUNK_MESH_VERTICES_NUM as usize];
        let mut indices = vec![0u32; CHUNK_MESH_INDICES_NUM as usize];

        
        reader.read_exact(bytemuck::cast_slice_mut(&mut vertices))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut indices))?;
        Ok(Self{
            mesh: mesh::Mesh{
                vertices,
                indices,
            }
        })
    }
}



pub struct TerrainChunkState{
    mesh_state: mesh::MeshState,
}

impl TerrainChunkState{
    
    pub fn from_chunk(chunk: &TerrainChunk, gpu_agent: &gpu::GpuAgent) -> Self{
        Self{
            mesh_state: mesh::MeshState::new(gpu_agent, &chunk.mesh),
        }
        
    }
}

pub struct PlanetState<'a>{
    agent: &'a gpu::GpuAgent,
    terrain_render_pipeline: wgpu::RenderPipeline,
    sea_render_pipeline: wgpu::RenderPipeline,
    atomsphere_render_pipeline: wgpu::RenderPipeline,
    terrain_chunk_states_cache: RefCell<lru::LruCache<(u8, u32), TerrainChunkState>>,

    terrain_chunk_states: Vec<&'a TerrainChunkState>,
}

impl<'b, 'a: 'b> PlanetState<'a> {
    const POOL_MAX:u32 = 10000;

    pub fn new(agent: &'a gpu::GpuAgent) -> Self{
        use mesh::VertexLayout;
        let terrain_vertex_layouts = mesh::MeshVertex::vertex_layout::<0>();
        let terrain_pipeline_layout = agent.create_pipeline_layout(&[], &[camera::Camera::PUSH_CONSTANT_RANGE], "terrain pipeline layout");

        let terrain_shader = agent.create_shader(include_str!("shaders/terrain.wgsl"), "terrain shader");

        let terrain_render_pipeline = agent.create_render_pipeline(
            &terrain_pipeline_layout, 
            &[terrain_vertex_layouts],
            &terrain_shader, 
            wgpu::PolygonMode::Fill,
            agent.config.format, 
            texture::Texture::DEPTH_FORMAT, 
            Some(wgpu::Face::Back),
            "terrain render pipeline"
        );

        let sea_pipeline_layout = agent.create_pipeline_layout(&[], &[camera::Camera::PUSH_CONSTANT_RANGE], "sea pipeline layout");

        let sea_shader = agent.create_shader_from_path("shaders/sea.wgsl");
        let sea_render_pipeline = agent.create_render_pipeline(
            &sea_pipeline_layout,
            &[], 
            &sea_shader, 
            wgpu::PolygonMode::Fill, 
            agent.config.format, 
            texture::Texture::DEPTH_FORMAT, 
            Some(wgpu::Face::Back),
            "sea render pipeline"
        );

        let atomsphere_pipeline_layout = agent.create_pipeline_layout(&[], &[camera::Camera::PUSH_CONSTANT_RANGE], "atomsphere pipeline layout");

        let atomsphere_shader = agent.create_shader(include_str!("shaders/atomsphere.wgsl"), "atomsphere shader");
        let atomsphere_render_pipeline = agent.create_render_pipeline(
            &atomsphere_pipeline_layout, 
            &[], 
            &atomsphere_shader, 
            wgpu::PolygonMode::Fill, 
            agent.config.format, 
            texture::Texture::DEPTH_FORMAT, 
            Some(wgpu::Face::Back),
            "atomsphere render pipeline"
        );
        
        Self{
            terrain_render_pipeline,
            sea_render_pipeline,
            atomsphere_render_pipeline,
            agent,
            terrain_chunk_states_cache: RefCell::new( lru::LruCache::new(std::num::NonZeroUsize::new(Self::POOL_MAX as usize).unwrap())),
            terrain_chunk_states: vec![],
        }
    }

    

    pub fn update(&mut self, planet: & Planet){
        for fi in 0..planet.lod_trees.len(){
            let tree = &planet.lod_trees[fi];
            for node in tree.get_leaves(){
                let head_index = node.terrain_chunk_index;
                let index = (fi as u8, head_index);
                if !self.terrain_chunk_states_cache.borrow().contains(&index){
                    let mut chunk_cache = planet.terrain_chunk_cache.borrow_mut();
                    let chunk = chunk_cache.get(&(fi as u8, head_index as u32)).unwrap();
                    let chunk_state = TerrainChunkState::from_chunk(chunk, self.agent);
                    self.terrain_chunk_states_cache.borrow_mut().put(index, chunk_state);
                }
            }
        }

    

    }

    pub fn render_terrain(
        &'a self, 
        render_pass: &'b mut wgpu::RenderPass<'b>, 
        camera: &camera::Camera,
        planet: &mut Planet,
    ) {
        render_pass.set_pipeline(&self.terrain_render_pipeline);
        
        for chunk_state in self.terrain_chunk_states.iter() {
            render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, camera.get_uniform_data() );
            render_pass.set_vertex_buffer(0, chunk_state.mesh_state.vertex_buffer.slice(..));
            render_pass.set_index_buffer(chunk_state.mesh_state.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk_state.mesh_state.index_count, 0, 0..1);
        }
    }

    
}







const CHUNK_MESH_SUBDIVISIONS:u32 = 16;
const CHUNK_MESH_VERTICES_NUM:u32 = (CHUNK_MESH_SUBDIVISIONS +1) * (CHUNK_MESH_SUBDIVISIONS +1);
const CHUNK_MESH_INDICES_NUM:u32 = CHUNK_MESH_SUBDIVISIONS * CHUNK_MESH_SUBDIVISIONS * 6;
const CHUNK_MESH_DATA_BYTE_NUM:u32 = CHUNK_MESH_VERTICES_NUM * std::mem::size_of::<mesh::MeshVertex>() as u32 + CHUNK_MESH_INDICES_NUM * std::mem::size_of::<u32>() as u32;



pub struct Planet{
    
    pub name: Name,
    pub radius: f32,
    pub lod_level: u8,
    pub position: Vec3<f32>,
    pub rotation: Quaternion<f32>,
    pub terrain_chunk_cache: RefCell<lru::LruCache<(u8, u32), TerrainChunk>>,

    terrain_chunk_heads: [Vec<TerrainChunkHead>; 6],
    lod_trees: [LodTree; 6],
    ron_file: std::fs::File,
    terrain_data_file: std::fs::File,
}

#[derive(Serialize, Deserialize)]
pub struct PlanetDescriptor{
    pub name: Name,
    pub radius: f32,
    pub lod_level: u8,
    pub position: Vec3<f32>,
    pub rotation: Quaternion<f32>,
}





struct LodTree{
    root: ChunkNode,
    lod_level: u8,
}

impl LodTree{
    pub fn new(radius: f32, center: Vec3<f32>, axis: math::AxisNormal, lod_level:u8) -> Self{
        let corners:[[f32; 3]; 4] = mesh::QUAD_CORNERS.map(|(x, y)| {
            
            let p = Vec3::from(axis.normal()) + Vec3::from( axis.tangent()) * x + Vec3::from(axis.btangent()) * y;
            let corner = center + (p.normalized() * radius);

            corner.into_array()
        });
        let root = ChunkNode{
            terrain_chunk_index: 0,
            children: None,
        };


        Self{
            root,
            lod_level,
        }
    }

    pub fn get_leaves(&self) -> Vec<&ChunkNode> {
        let mut leaves = Vec::new();
        let mut stack = vec![&self.root];
        while let Some(node) = stack.pop(){
            if let Some(children) = &node.children{
                stack.extend(children.iter().map(|child| child.as_ref()));
            }else{
                leaves.push(node);
            }
        }
        leaves
    }
}




impl Planet{
    const POOL_MAX: u32 = 10000;
    
    ///create new planet and save
    /// save directory structure:
    /// world_path/
    ///     planet_name/
    ///         planet_name.ron
    ///         planet_name.terrain
    pub fn new(desc: &PlanetDescriptor, world_dir: std::path::PathBuf) -> Result<Self, std::io::Error>{

        let ron_file_path = world_dir.join(desc.name).with_extension("ron");
        let ron_file = utils::create_new_file(ron_file_path)?;
        let mut ron_writer = std::io::BufWriter::new(ron_file);
        ron_writer.write_all(ron::ser::to_string(&desc).unwrap().as_bytes())?;
        let ron_file = ron_writer.into_inner().unwrap();

        let terrain_data_file_path = world_dir.join(desc.name).with_extension("terrain");
        let terrain_data_file = utils::create_new_file(terrain_data_file_path)?;
        let mut terrain_data_writer = std::io::BufWriter::new(terrain_data_file);



        let rotation = Quaternion::<f32>::identity();
        let face_chunks_num = 4u32.pow(desc.lod_level as u32 +1) -1;
        
        
        //generate terrain chunks head and save chunk data to file
        let mut terrain_chunk_heads: [Vec<TerrainChunkHead>; 6] = Default::default();
        for (face_index, &axis) in math::AxisNormal::AXIS_ARRAY.iter().enumerate()  {
            let mut level_left_index = 1;
            let mut subdivision = 1;
            let mut depth = 0;
            let mut detail_value = 2.0f32.powi(depth as i32 - desc.lod_level as i32);

            let mut chunk_heads: Vec<TerrainChunkHead> = Vec::new();

            for i in 0..face_chunks_num {
                if i >= level_left_index{
                    level_left_index = 4*level_left_index + 1;
                    depth += 1;
                    detail_value *= 2.0;
                    subdivision *= 2;
                }

                let level_index = i - level_left_index;

                let (xi, yi) = (level_index % subdivision , level_index / subdivision);
            
                //unit cube
                
                let d: f32 =  2.0 / subdivision as f32;
                let start = [
                    -1.0 + (xi as f32) * d,
                    -1.0 + (yi as f32) * d
                ].into();
                let size: f32 = 2.0/subdivision as f32;
                let offset: f32 = 1.0;
                
                let mut mesh = mesh::Mesh::create_axis_normal_terrain_plane(
                    axis, start, size, offset, CHUNK_MESH_SUBDIVISIONS);

                //spherelize
                for vertex in mesh.vertices.iter_mut() {
                    let p = Vec3::<f32>::from(vertex.pos)
                    .map(|s|{ (s*std::f32::consts::PI / 4.0).tan()})
                    .normalized();
                    
                    vertex.normal = p.into_array();
                    vertex.pos = (p * desc.radius).into_array();
                }
                
                
                let terrain_chunk = TerrainChunk{
                    mesh
                };
                //write terrain chunk
                terrain_chunk.to_writer(&mut terrain_data_writer).unwrap();

                let raw_corners = mesh::QUAD_CORNERS.map(|(x, y)|{
                    let xy = [x, y].map(|s| (s+1.0)/2.0 * d);
                    let c = [xy[0]+ start.x, xy[1]+start.y];
                    c
                });

                let corners = raw_corners.map(|[x, y]|{
                    let c = (Vec3::<f32>::from(axis.normal()) + Vec3::from(axis.tangent())*x + Vec3::from(axis.btangent())*y)
                    .map(|s| (s*std::f32::consts::PI / 4.0).tan()).normalized() * desc.radius;
                    c.into_array()
                });

                let chunk_head =  TerrainChunkHead{
                    depth,
                    index: i,
                    axis_normal: axis,
                    detail_value,
                    raw_corners,
                    corners,
                    subdivision,
                };

                chunk_heads.push(chunk_head);
            }


            terrain_chunk_heads[face_index] = chunk_heads;
            
        }


        let lod_trees:[LodTree; 6] = math::AxisNormal::AXIS_ARRAY.map(|i| LodTree::new(desc.radius, desc.position, i, desc.lod_level) );

        let terrain_chunk_cache = RefCell::new( lru::LruCache::new(std::num::NonZeroUsize::new(Self::POOL_MAX as usize).unwrap()));

        let terrain_data_file = terrain_data_writer.into_inner().unwrap();
        Ok(
            Self{
                name: desc.name.clone(),
                position: desc.position,
                terrain_chunk_heads,
                rotation,
                radius: desc.radius,
                lod_trees,
                lod_level:desc.lod_level,
                terrain_chunk_cache,

                ron_file,
                terrain_data_file,
            }
        )
    }

    
    pub fn load_chunk_data(&self, tree_index: usize, chunk_index: usize) -> Result<TerrainChunk, std::io::Error>{
        let mut reader = std::io::BufReader::new(&self.terrain_data_file);
        let chunks_num_per_face = self.terrain_chunk_heads[tree_index].len();
        let data_index = tree_index * chunks_num_per_face + chunk_index;
        reader.seek(std::io::SeekFrom::Start(data_index as u64 * CHUNK_MESH_DATA_BYTE_NUM as u64));
        TerrainChunk::from_reader(&mut reader)
        
    }

    pub fn update(&mut self, delta_time: f32,  camera: &camera::Camera){
        self.update_lod_trees(camera);
    }


    pub fn update_lod_trees(&mut self, camera: &camera::Camera){
        for (index ,tree) in self.lod_trees.iter_mut().enumerate() {

            let mut stack: Vec<&mut ChunkNode> = vec![];
            stack.push(&mut tree.root);
            while let Some(node) = stack.pop() {// dfs lod tree
                let terrain_chunk_head = &self.terrain_chunk_heads[index as usize][node.terrain_chunk_index as usize];
                let detail_value = terrain_chunk_head.detail_value;
                let calc_detail = terrain_chunk_head.calc_detail(camera, self.position);



                if calc_detail >= detail_value{
                    let temp = &mut node.children;
                    match temp {
                        Some(children) => {
                            for child in children.iter_mut() {
                                stack.push(child);
                            }
                            
                        }
                        None => {
                            let children:[Box<ChunkNode>; 4] = [0, 1, 2, 3].map(|i| {
                                let chunk_node = ChunkNode{
                                    terrain_chunk_index: 4 * node.terrain_chunk_index + i + 1, 
                                    children: None,
                                };
                                Box::new(chunk_node)
                            });
                            //add stack
                            std::mem::replace(temp, Some(children));
                            
                            for child in temp.as_mut().unwrap().iter_mut() {
                                stack.push(child);
                            }
                            
                        }
                    }
                }
                else {
                    if node.children.is_some() {
                        node.children = None;
                    }
                    
                }
            }
        }
        
    }

    pub fn description(&self) ->PlanetDescriptor {
        PlanetDescriptor{
            name: self.name.clone(),
            position: self.position,
            radius: self.radius,
            lod_level: self.lod_level,
            rotation: self.rotation,
        }
    }

    pub fn save(&self) {
        let mut writer = std::io::BufWriter::new(&self.ron_file);
        let desc = self.description();
        ron::ser::to_writer_pretty(&mut writer, &desc, ron::ser::PrettyConfig::default()).unwrap();
    }
 
  
}



