
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::io::SeekFrom;
use crate::camera;
use crate::camera::Camera;
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




pub struct ChunkInfo{
    pub raw_corners: [[f32; 2];4],
    pub axis_normal: math::AxisNormal,
    pub corners: [[f32; 3]; 4],
    pub depth: u8,
    pub detail_value: f32,
    pub subdivision: u32,


}

pub struct PlotInfo{
    pub position: Vec3<f32>,
    pub axis_normal: math::AxisNormal,
    pub grid_coord: [u32; 2],
}

impl ChunkInfo{
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


pub struct ChunkTerrainData{
    mesh: mesh::Mesh,
}

impl ChunkTerrainData{
    pub const CHUNK_TERRAIN_DATA_SIZE:u64 = std::mem::size_of::<Self>() as u64;

    pub fn from_plot_data(plot_data: &PlotTerrainData, plot_info: &PlotInfo, ) -> Self{
        let triangles = mesh::Triangles::create_grid_on_unit_cube(index_x, index_y, axis, segment_num, grid_segment_num)
        let mesh = mesh::Mesh::
    }

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



pub struct ChunkTerrainState{
    mesh_state: mesh::MeshState,
}

impl ChunkTerrainState{
    
    pub fn from_chunk(chunk: &ChunkTerrainData, gpu_agent: &gpu::GpuAgent) -> Self{
        Self{
            mesh_state: mesh::MeshState::new(gpu_agent, &chunk.mesh),
        }
        
    }
}

pub struct PlotTerrainData{
    pub cell_positions: Vec<Vec3<f32>>,
    pub cell_elevations: Vec<f32>,
    pub cell_normals: Vec<Vec3<f32>>,
}

impl PlotTerrainData{
    pub fn new(cell_positions: Vec<Vec3<f32>>, map_func: impl Fn(&Vec3<f32>) -> (f32, Vec3<f32>)) -> Self{
        let cell_elevations = cell_positions.iter().map(|pos|{
            map_func(pos).0
        }).collect();
        let cell_normals = cell_positions.iter().map(|pos|{
            map_func(pos).1
        }).collect();

        Self{
            cell_positions,
            cell_elevations,
            cell_normals,
        }
    }
}

pub struct Region{
    pub axis: math::AxisNormal,
    pub plot_positions: Vec<Vec3<f32>>,
    

    pub plot_details: Vec<f32>,
    pub chunk_details: Vec<f32>,

    pub terrain_chunk_cache: utils::ArrayPool<ChunkTerrainData>,

}

pub struct RegionsShareInfo{
    pub lod_level: u8,
    pub region_side_plots_num: u32,
    pub region_plots_num: u32,
    pub region_chunks_num: u32,
}

impl RegionsShareInfo{
    pub fn new(lod_level: u8) -> Self{
        let region_side_plots_num = 2u32.pow(lod_level as u32);
        let region_plots_num = region_side_plots_num.pow(2);
        let region_chunks_num = (region_plots_num*4 -1) / 3;
        Self{
            lod_level,
            region_side_plots_num,
            region_plots_num,
            region_chunks_num,
        }
    }
}

//one of six faces of a cube-sphere planet
impl Region {
    const POOL_MAX:u32 = 100;

    pub fn new(axis: math::AxisNormal, planet_desc: &PlanetDescriptor) -> Self{
        //could be optimized
        let region_side_plots_num = 2u32.pow(planet_desc.lod_level as u32);
        let region_plots_num = region_side_plots_num.pow(2);
        let region_chunks_num = (region_plots_num*4 -1) / 3;
        

        let mut plot_positions = Vec::new();
        for yi in 0..region_side_plots_num {
            for xi in 0..region_side_plots_num{
                let plot_index = yi * region_side_plots_num + xi;
                
                let pos = axis.mat3() * Vec3::new(
                    -1.0 + (xi as f32 + 0.5) * 2.0 / region_side_plots_num as f32,
                    -1.0 + (yi as f32 + 0.5) * 2.0 / region_side_plots_num as f32,
                    1.0
                );
                
                //cobe wrap pos
                let mut pos = pos.into_array();
                math::cobe_wrap_with_axis(&mut pos, axis);
                let pos = Vec3::<f32>::from(pos).normalized() * planet_desc.radius;

                plot_positions[plot_index as usize] = pos;

            };
        };

        let plot_details = vec![0.0f32; region_plots_num as usize];

        let chunk_details = vec![0.0f32; region_chunks_num as usize];

        let terrain_chunk_cache = utils::ArrayPool::new(Self::POOL_MAX, region_chunks_num as usize);

        Self { 
            axis,
            plot_positions,
            plot_details,
            chunk_details,
            terrain_chunk_cache,
        }
    }

    pub fn calc_detail_value(camera: &camera::Camera, pos: &Vec3<f32>, normal: &Vec3<f32>) -> f32{
        let (dotre, distance) = camera.ray_dot(*pos, *normal);
        let detail_value = if dotre < 0.0{
            0.0f32
        } else{
            
            dotre * 1.0 / (1.0 + distance/(camera.projection.zfar/2.0))
        };
        detail_value
    }

    

    pub fn update(&mut self, camera: &camera::Camera){

        for i in 0..self.plot_positions.len(){
            let pos = self.plot_positions[i];
            let detail_value = Self::calc_detail_value(camera, &pos, &pos.normalized());
            self.plot_details[i] = detail_value;
        }
        let chunk_len = self.chunk_details.len();
        let plot_len = self.plot_details.len();

        for i in 0..plot_len {
            self.chunk_details[chunk_len - plot_len + i] = self.plot_details[i];
        }

        for i in (0..chunk_len-plot_len).rev(){
            self.chunk_details[i] = 
                self.chunk_details[i *4 +1] + self.chunk_details[i*4 +2] + self.chunk_details[i *4 +3] + self.chunk_details[i*4 +4];
        }

        
    }

    fn update_terrain<R>(&mut self, terrain_datat_reader:&mut R) where R: std::io::Read, R: std::io::Seek{
        for i in 0..self.chunk_details.len(){
            let chunk_detail = self.chunk_details[i];
            
            if chunk_detail >= 1.0 && self.terrain_chunk_cache.get(i as u32).is_none(){
                terrain_datat_reader.seek(SeekFrom::Start(i as u64 * ChunkTerrainData::CHUNK_TERRAIN_DATA_SIZE));
                let terrain_data = ChunkTerrainData::from_reader(terrain_datat_reader).unwrap();

                self.terrain_chunk_cache.put(i as u32, terrain_data);
            }
        }
    }




}

pub struct RegionState{

    
    pub terrain_chunk_states_cache: utils::ArrayPool<ChunkTerrainState>,
}

impl RegionState{
    pub fn from_region(region: &Region, gpu_agent: &gpu::GpuAgent, region_share: &RegionsShareInfo) -> Self{
        let terrain_chunk_states_cache = utils::ArrayPool::new(Region::POOL_MAX, region_share.region_chunks_num as usize);
        Self{
            terrain_chunk_states_cache,
        }
    }

    pub fn update(&mut self, region: &Region, gpu_agent: &gpu::GpuAgent, region_share: &RegionsShareInfo){
        for i in 0..region_share.region_plots_num {
            let chunk_detail = region.chunk_details[i as usize];
            
            
            if chunk_detail >= 1.0{
                let chunk = region.terrain_chunk_cache.get(i as u32).unwrap();
                let chunk_state = ChunkTerrainState::from_chunk(chunk, gpu_agent);
                self.terrain_chunk_states_cache.put(i as u32, chunk_state);
            }
        }
    }

    pub fn get_visible_terrain_states(&self, region: &Region, region_share: &RegionsShareInfo) -> Vec<&ChunkTerrainState>{
        let mut visible = Vec::new();
        for i in 0..region_share.region_plots_num {
            let chunk_detail = region.chunk_details[i as usize];
            
            
            if chunk_detail >= 1.0{
                let chunk_state = self.terrain_chunk_states_cache.get(i as u32).unwrap();
                visible.push(chunk_state);
            }
        }
        visible
    }


}

pub struct PlanetState<'a>{
    agent: &'a gpu::GpuAgent,
    terrain_render_pipeline: wgpu::RenderPipeline,
    sea_render_pipeline: wgpu::RenderPipeline,
    atomsphere_render_pipeline: wgpu::RenderPipeline,
    region_states: [RegionState; 6],
}

impl<'b, 'a: 'b> PlanetState<'a> {
    const POOL_MAX:u32 = 10000;

    pub fn new(agent: &'a gpu::GpuAgent, planet: & Planet) -> Self{
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

        let region_states = [
            RegionState::from_region(&planet.regions[0], agent, &planet.regions_share),
            RegionState::from_region(&planet.regions[1], agent, &planet.regions_share),
            RegionState::from_region(&planet.regions[2], agent, &planet.regions_share),
            RegionState::from_region(&planet.regions[3], agent, &planet.regions_share),
            RegionState::from_region(&planet.regions[4], agent, &planet.regions_share),
            RegionState::from_region(&planet.regions[5], agent, &planet.regions_share),
        ];
        
        Self{
            terrain_render_pipeline,
            sea_render_pipeline,
            atomsphere_render_pipeline,
            agent,
            region_states,
        }
    }

    

    pub fn update(&mut self, planet: & Planet){
        
        for ri in 0..6{
            self.region_states[ri].update(&planet.regions[ri], self.agent, &planet.regions_share);
        }

    }

    pub fn render_terrain(
        &'a self, 
        render_pass: &'b mut wgpu::RenderPass<'b>, 
        camera: &camera::Camera,
        planet: &mut Planet,
    ) {
        render_pass.set_pipeline(&self.terrain_render_pipeline);

        for ri in 0..6{
            
            let terrain_states = self.region_states[ri].get_visible_terrain_states(&planet.regions[ri], &planet.regions_share);

            for chunk_state in terrain_states.iter() {
                render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, camera.get_uniform_data() );
                render_pass.set_vertex_buffer(0, chunk_state.mesh_state.vertex_buffer.slice(..));
                render_pass.set_index_buffer(chunk_state.mesh_state.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk_state.mesh_state.index_count, 0, 0..1);
            }
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

    pub regions: [Region; 6],
    pub regions_share: RegionsShareInfo,

    
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







impl Planet{
    const POOL_MAX: u32 = 10000;

    pub fn build(desc: &PlanetDescriptor, world_dir: std::path::PathBuf) -> Result<(), std::io::Error>{
        let ron_file_path = world_dir.join(desc.name).with_extension("ron");
        let ron_file = utils::create_new_file(ron_file_path)?;
        let mut ron_writer = std::io::BufWriter::new(ron_file);
        ron_writer.write_all(ron::ser::to_string(&desc).unwrap().as_bytes())?;
        let ron_file = ron_writer.into_inner().unwrap();

        let terrain_data_file_path = world_dir.join(desc.name).with_extension("terrain");
        let terrain_data_file = utils::create_new_file(terrain_data_file_path)?;
        let mut terrain_data_writer = std::io::BufWriter::new(terrain_data_file);
        
        {//create terrain data

            let info = RegionsShareInfo::new(desc.lod_level);

            for ri in 0..6{
                let plot_terrain 
            }
        };
        

    }
    
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



        let rotation: Quaternion<f32> = Quaternion::<f32>::identity();
        let region_chunks_num = 4u32.pow(desc.lod_level as u32 +1) -1;
        let region_plot_num = 4u32.pow(desc.lod_level as u32);
        let region_side_plot_num = 2u32.pow(desc.lod_level as u32);
        
        //generate terrain chunks head and save chunk data to file
        let mut terrain_chunk_heads: [Vec<ChunkInfo>; 6] = Default::default();

        for (face_index, &axis) in math::AxisNormal::AXIS_ARRAY.iter().enumerate()  {
            let mut level_left_index = 1;
            let mut subdivision = 1;
            let mut depth = 0;
            let mut detail_value = 2.0f32.powi(depth as i32 - desc.lod_level as i32);

            let mut chunk_heads: Vec<ChunkInfo> = Vec::new();

            for i in 0..region_chunks_num {
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
                
                
                let terrain_chunk = ChunkTerrainData{
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

                let chunk_head =  ChunkInfo{
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

        let plot_positions = {
            let mut plot_pos: [Vec<Vec3<f32>>; 6] = math::AxisNormal::AXIS_ARRAY.map(|i| vec![Vec3::<f32>::zero(); region_plot_num as usize]);

            //fill plot pos array 
            for axis in math::AxisNormal::AXIS_ARRAY{
                for yi in 0..region_side_plot_num {
                    for xi in 0..region_side_plot_num{
                        let plot_index = yi * region_side_plot_num + xi;
                        
                        let pos = axis.mat3() * Vec3::new(
                            -1.0 + (xi as f32 + 0.5) * 2.0 / region_side_plot_num as f32,
                            -1.0 + (yi as f32 + 0.5) * 2.0 / region_side_plot_num as f32,
                            1.0
                        );
                        
                        //cobe wrap pos
                        let mut pos = pos.into_array();
                        math::cobe_wrap_with_axis(&mut pos, axis);
                        let pos = Vec3::<f32>::from(pos).normalized() * desc.radius;

                        plot_pos[axis.index()][plot_index as usize] = pos;

                    }
                }
            }
            plot_pos
        };

        let mut plot_details: [Vec<f32>; 6] = math::AxisNormal::AXIS_ARRAY.map(|i| vec![0.0; region_plot_num as usize]);

        
        
        

        

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

    
    pub fn load_chunk_data(&self, tree_index: usize, chunk_index: usize) -> Result<ChunkTerrainData, std::io::Error>{
        let mut reader = std::io::BufReader::new(&self.terrain_data_file);
        let chunks_num_per_face = self.terrain_chunk_heads[tree_index].len();
        let data_index = tree_index * chunks_num_per_face + chunk_index;
        reader.seek(std::io::SeekFrom::Start(data_index as u64 * CHUNK_MESH_DATA_BYTE_NUM as u64));
        ChunkTerrainData::from_reader(&mut reader)
        
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



