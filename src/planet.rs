
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs;
use std::io::SeekFrom;
use crate::camera;
use crate::camera::Camera;
use crate::math;
use crate::math::AxisNormal;
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

    pub fn from_plot_data(plot_data: &PlotTerrainData, plot_info: &PlotInfo, planet_desc: &PlanetDescriptor) -> Self{
        let segment_num = planet_desc.calc_plots_side_num() as u32;

        let triangles = mesh::Triangles::create_grid_on_unit_cube(
            plot_info.grid_coord[0],
            plot_info.grid_coord[1], 
            plot_info.axis_normal, 
            segment_num, 
            planet_desc.mesh_grid_segment_num,
        );

        // let mesh = mesh::Mesh::from_triangles(&triangles);

        let vertices = triangles.0.iter().map(|v|{

            let position: Vec3<f32> = (*v).into();
            let xy = plot_info.axis_normal.get_xy(v);
            let uv = Vec2::new((xy[0] + 1.0) / segment_num as f32, (xy[1] + 1.0) / segment_num as f32);


            let normal = position.normalized();
            let position = position.normalized() * planet_desc.radius;

            

        
        

            mesh::MeshVertex{
                pos: position.into_array(),
                normal: normal.into_array(),
                tex_coord: uv.into_array(),
            }
        }).collect();


        let mesh = mesh::Mesh{
            vertices,
            indices: triangles.1,
        };

        Self{
            mesh,
        }
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
    pub cell_positions: utils::Grid<[f32; 3]>,
    pub cell_elevations: utils::Grid<f32>,
    pub cell_normals: utils::Grid<[f32; 3]>,
}

impl PlotTerrainData{
    pub const ELEMENT_BYTE_SIZE: u64 = std::mem::size_of::<([f32; 7])>() as u64;

    pub fn new(cell_positions: utils::Grid<[f32; 3]>, map_func: impl Fn(&[f32; 3]) -> (f32, [f32; 3])) -> Self{
        let cell_elevations: utils::Grid<f32> = cell_positions.map(|pos|{
            map_func(pos).0
        });

        let cell_normals = cell_positions.map(|pos|{
            map_func(pos).1
        });

        Self{
            cell_positions,
            cell_elevations,
            cell_normals,
        }
    }

    pub fn to_writer<W>(&self, writer: &mut W) -> Result<(), std::io::Error> where W: std::io::Write{
        writer.write_all(bytemuck::cast_slice(&self.cell_positions.data))?;
        writer.write_all(bytemuck::cast_slice(&self.cell_elevations.data))?;
        writer.write_all(bytemuck::cast_slice(&self.cell_normals.data))?;
        Ok(())
    }

    //create from reader with specified element num
    pub fn from_reader<R>(reader: &mut R, cells_side_num: u32) -> Result<Self, std::io::Error> where R: std::io::Read{
        let element_num = cells_side_num * cells_side_num;
        let mut cell_positions = vec![[0.0f32; 3]; element_num as usize];
        let mut cell_elevations = vec![0.0f32; element_num as usize];
        let mut cell_normals = vec![[0.0f32; 3]; element_num as usize];

        reader.read_exact(bytemuck::cast_slice_mut(&mut cell_positions))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut cell_elevations))?;
        reader.read_exact(bytemuck::cast_slice_mut(&mut cell_normals))?;
        Ok(Self{
            cell_positions: utils::Grid::new_square(cells_side_num, cell_positions),
            cell_elevations: utils::Grid::new_square(cells_side_num, cell_elevations),
            cell_normals: utils::Grid::new_square(cells_side_num, cell_normals),
        })
    }

    
}

pub struct Region{
    pub axis: math::AxisNormal,
    pub plot_positions_grid: utils::Grid<[f32; 3]>,
    
    pub plot_datas_grid: utils::Grid<PlotTerrainData>,

    //vary
    pub plot_details_grid: utils::Grid<f32>,
    pub chunk_details: Vec<f32>,
    pub terrain_chunk_cache: utils::ArrayPool<ChunkTerrainData>,

}

#[derive(Serialize, Deserialize)]
struct RegionRon{
    pub axis: math::AxisNormal,
    pub plot_positions_grid: utils::Grid<[f32; 3]>,
}

pub struct RegionsShareInfo{
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
            region_side_plots_num,
            region_plots_num,
            region_chunks_num,
        }
    }
}

//one of six faces of a cube-sphere planet
impl Region {
    const POOL_MAX:u32 = 100;

    pub fn build(axis: math::AxisNormal, planet_desc: &PlanetDescriptor, noise_func: impl Fn(&[f32; 3]) -> (f32, [f32; 3]),planet_dir: std::path::PathBuf) {
        let region_side_plots_num = 2u32.pow(planet_desc.lod_level as u32);
        let region_plots_num = region_side_plots_num.pow(2);
        let region_chunks_num = (region_plots_num*4 -1) / 3;

       
        
        //creat a plot_data file
        let plot_data_file_path = planet_dir.join(format!("plot_datas_{}.bin", axis));
        let mut plot_data_file = std::fs::File::create(plot_data_file_path).unwrap();
        let mut plot_data_writer = std::io::BufWriter::new(&mut plot_data_file);
        
        //create a chunk terrain data file
        let chunk_data_file_path = planet_dir.join(format!("chunk_datas_{}.bin", axis));
        let mut chunk_data_file = std::fs::File::create(chunk_data_file_path).unwrap();
        let mut chunk_data_writer = std::io::BufWriter::new(&mut chunk_data_file);

        let mut plot_positions:Vec<[f32; 3]> = vec![[0.0,0.0,0.0]; region_plots_num as usize];
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

                let plot_info = PlotInfo{
                    position: pos,
                    axis_normal: axis,
                    grid_coord: [xi, yi],
                };

                let cell_positions = mesh::Triangles::create_grid_points_on_unit_cube(xi, yi, axis, region_side_plots_num, planet_desc.mesh_grid_segment_num).into_iter().map(|p| {
                    let p = Vec3::<f32>::from(p);
                    let p = p.normalized() * planet_desc.radius;
                    p.into_array()
                }).collect();
                let cell_positions = utils::Grid::new_square(planet_desc.mesh_grid_segment_num, cell_positions);
                

                let plot_data = PlotTerrainData::new(cell_positions, &noise_func);
                plot_data.to_writer(&mut plot_data_writer).unwrap();

                let chunk_terrain_data = ChunkTerrainData::from_plot_data(&plot_data, &plot_info, planet_desc);

                chunk_terrain_data.to_writer(&mut chunk_data_writer).unwrap();

                plot_positions[plot_index as usize] = pos.into_array();

            };

            
        };

        
         //create a region ron file
        let region_ron_file_path = planet_dir.join(format!("region_{}.ron", axis));
        let mut region_ron_file = std::fs::File::create(region_ron_file_path).unwrap();
        let mut region_ron_writer = std::io::BufWriter::new(&mut region_ron_file);

        let region_ron = RegionRon{
            axis,
            plot_positions_grid: utils::Grid::new_square(region_side_plots_num, plot_positions),
        };

        ron::ser::to_writer_pretty(&mut region_ron_writer, &region_ron, ron::ser::PrettyConfig::default()).unwrap();
        
    }

  


    pub fn load_from_dir(axis: math::AxisNormal, planet_dir: std::path::PathBuf, region_share: &RegionsShareInfo, planet_desc: &PlanetDescriptor) -> Result<Self, std::io::Error>{

        

        let plot_details = utils::Grid::new_square_with_default(region_share.region_side_plots_num as u32, 0.0f32);

        let chunk_details = vec![0.0f32; region_share.region_chunks_num as usize];

        let terrain_chunk_cache = utils::ArrayPool::<ChunkTerrainData>::new(Self::POOL_MAX, region_share.region_chunks_num as usize);

        //load regionron from file
        let region_ron: RegionRon = {
            let region_ron_file_path = planet_dir.join(format!("region_{}.ron", axis));
            let region_ron_file = std::fs::File::open(region_ron_file_path)?;
            let region_ron_reader = std::io::BufReader::new(region_ron_file);
            ron::de::from_reader(region_ron_reader).unwrap()
        };

        //load plot datas from file
        let plot_datas: Vec<PlotTerrainData> = {
            let plot_data_file_path = planet_dir.join(format!("plot_datas_{}.bin", axis));
            let plot_data_file = std::fs::File::open(plot_data_file_path)?;
            let mut plot_data_reader = std::io::BufReader::new(plot_data_file);
            let mut plot_datas = Vec::new();

            let cells_side_num = planet_desc.mesh_grid_segment_num;
            for _ in 0..region_share.region_plots_num{
                let plot_data = PlotTerrainData::from_reader(&mut plot_data_reader, cells_side_num).unwrap();
                plot_datas.push(plot_data);
            }
            plot_datas
        };

        //return self

        Ok(Self{
            axis,
            plot_positions_grid: region_ron.plot_positions_grid,
            plot_details_grid: plot_details,
            chunk_details,
            plot_datas_grid: utils::Grid::new_square(region_share.region_side_plots_num as u32, plot_datas),
            terrain_chunk_cache,
        })

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

        for i in 0..self.plot_positions_grid.len(){
            let pos:Vec3<f32> =  self.plot_positions_grid[i].into();
            let detail_value = Self::calc_detail_value(camera, &pos, &pos.normalized());
            self.plot_details_grid[i] = detail_value;
        }
        let chunk_len = self.chunk_details.len();
        let plot_len = self.plot_details_grid.len();

        for i in 0..plot_len {
            self.chunk_details[chunk_len - plot_len + i] = self.plot_details_grid[i];
        }

        for i in (0..chunk_len-plot_len).rev(){
            self.chunk_details[i] = 
                self.chunk_details[i *4 +1] + self.chunk_details[i*4 +2] + self.chunk_details[i *4 +3] + self.chunk_details[i*4 +4];
        }

        
    }

    fn update_terrain<R>(&mut self, terrain_data_reader:&mut R) where R: std::io::Read, R: std::io::Seek{
        for i in 0..self.chunk_details.len(){
            let chunk_detail = self.chunk_details[i];
            
            if chunk_detail >= 1.0 && self.terrain_chunk_cache.get(i as u32).is_none(){
                terrain_data_reader.seek(SeekFrom::Start(i as u64 * ChunkTerrainData::CHUNK_TERRAIN_DATA_SIZE));
                let terrain_data = ChunkTerrainData::from_reader(terrain_data_reader).unwrap();

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

    pub plots_side_num: u32,
    pub mesh_grid_segment_num: u32,
}

#[derive(Serialize, Deserialize)]
pub struct PlanetDescriptor{
    pub name: Name,
    pub radius: f32,
    pub lod_level: u8,
    pub position: Vec3<f32>,
    pub rotation: Quaternion<f32>,
    pub mesh_grid_segment_num: u32,
    
}

impl PlanetDescriptor{
    pub fn calc_plots_side_num(&self) -> u32{
        2u32.pow(self.lod_level as u32)
    }
}





impl Planet{
    const POOL_MAX: u32 = 10000;
    
    pub fn build(planet_desc: &PlanetDescriptor, planet_dir: std::path::PathBuf) -> Result<(), std::io::Error>{
        
        {//descriptor ron file
            let ron_file_path = planet_dir.join(planet_desc.name).with_extension("ron");
            let ron_file = utils::create_new_file(ron_file_path)?;
            let mut ron_writer = std::io::BufWriter::new(ron_file);
            ron_writer.write_all(ron::ser::to_string(&planet_desc).unwrap().as_bytes())?;
        }

        
        

        {//build regions
            use noise::{NoiseFn, Perlin, Seedable};

            let perlin_noise = Perlin::new(1);
            let noise_func = |pos: &[f32; 3]| ->(f32, [f32; 3]) {
                let pos_f64 = pos.map(|s| s as f64);
                let elevation = perlin_noise.get(pos_f64);
                
                let pos = Vec3::<f32>::from(*pos);
                let normal = pos.normalized().into_array();
                (elevation as f32, normal)
            };
            let info = RegionsShareInfo::new(planet_desc.lod_level);
            let plot_side_num = planet_desc.calc_plots_side_num();
            for ri in 0..6{
                let axis = math::AxisNormal::from_u32(ri);
                Region::build(axis, planet_desc, noise_func, planet_dir.clone());

                
            }
        };
        
        Ok(())
    }

    pub fn load_from_dir(planet_dir: std::path::PathBuf, planet_name: utils::Name) -> Result<Self, std::io::Error>{
        //load descriptor from  exsited ron file
        

        let planet_desc: PlanetDescriptor = {
            let ron_file_path = planet_dir.join(planet_name.as_str()).with_extension("ron");
            let ron_file = fs::File::open(ron_file_path).unwrap();
            let mut ron_reader = std::io::BufReader::new(ron_file);
            ron::de::from_reader(&mut ron_reader).unwrap()
        };

        let region_share = RegionsShareInfo::new(planet_desc.lod_level);

        
        //load regions
        let mut regions = math::AxisNormal::AXIS_ARRAY.map(|axis| {
            Region::load_from_dir(axis, planet_dir.clone(), &region_share, &planet_desc).unwrap()
        });

        let plot_side_num = planet_desc.calc_plots_side_num();

        Ok(Self{
            name: planet_name,
            radius: planet_desc.radius,
            lod_level: planet_desc.lod_level,
            position: planet_desc.position,
            rotation: planet_desc.rotation,
            regions: regions,
            regions_share: region_share,
            plots_side_num: plot_side_num,
            mesh_grid_segment_num: planet_desc.mesh_grid_segment_num,
        })

    }
    
    
    pub fn description(&self) ->PlanetDescriptor {
        PlanetDescriptor{
            name: self.name.clone(),
            position: self.position,
            radius: self.radius,
            lod_level: self.lod_level,
            rotation: self.rotation,
            mesh_grid_segment_num: self.mesh_grid_segment_num,
        }
    }

    pub fn update(&mut self, delta_time: f32, camera: &camera::Camera){
        //todo!()
    }

    pub fn save(&self) {
        todo!()
    }
 
  
}




//test
#[cfg(test)]
mod test{
    use super::*;
    use crate::utils;
    use crate::math;
    use crate::mesh;
    use crate::camera;

    #[test]
    fn test_planet_build(){
        let planet_desc = PlanetDescriptor{
            name: utils::Name::new("AA"),
            radius: 100.0,
            lod_level: 4,
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
            mesh_grid_segment_num: 16,
        };
        let planet_dir = std::path::PathBuf::from("test/planet/AA");
        //create planet dir if not exsited
        if !planet_dir.exists(){
            fs::create_dir_all(&planet_dir).unwrap();
        }

        
        Planet::build(&planet_desc, planet_dir).unwrap();
    }

}