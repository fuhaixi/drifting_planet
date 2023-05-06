
use std::fs;
use std::io::SeekFrom;
use crate::camera;

use crate::math;

use crate::texture;
use crate::utils;
use serde::{Serialize, Deserialize};
use crate::gpu;
use crate::mesh;
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

   

    pub fn sample_on_grid_plot(plot_grid: &utils::Grid<PlotTerrainData>, region_uv: Vec2<f32>) -> (f32, [f32; 3]){
        assert!(region_uv.x >= 0.0 && region_uv.x <= 1.0);
        assert!(region_uv.y >= 0.0 && region_uv.y <= 1.0);

        let plot_side_cells_num = plot_grid.data[0].cell_elevations.extent.w as u32;
        
        let region_side_cells_num = plot_side_cells_num * plot_grid.extent.w as u32;
        let get_one = |coord: Vec2<i32>| ->(f32, [f32; 3]) {
            //clamp
            let coord = coord.map(|x|  {
                if x < 0 {
                    0
                } else if x >= region_side_cells_num as i32{
                    region_side_cells_num  - 1
                } else{
                    x as u32
                }
            });

            let plot_coord = coord.map(|x| x/ plot_side_cells_num );
            let cell_coord = coord.map(|x| x %  plot_side_cells_num );

            (
                plot_grid.get(plot_coord.into()).unwrap().cell_elevations.get(cell_coord.into()).unwrap().clone(),
                plot_grid.get(plot_coord.into()).unwrap().cell_normals.get(cell_coord.into()).unwrap().clone()
            )
            
        };

        let bilinear_sample = |uv: Vec2<f32>| ->(f32, [f32; 3]){
            let xy = uv.map(|x| x * region_side_cells_num as f32);
            let xy00 = xy.map(|x| x.floor() as i32 - 1i32 );
            let t = xy - xy00.map(|x| x as f32);
            let x_add:i32 = if xy00[0] +1 < region_side_cells_num as i32 {1} else {0};
            let y_add:i32 = if xy00[1] +1 < region_side_cells_num as i32 {1} else {0};
            let xy10 = xy00 + Vec2::new(x_add, 0);
            let xy01 = xy00 + Vec2::new(0, y_add);
            let xy11 = xy00 + Vec2::new(x_add, y_add);

            let (elevation00, normal00) = get_one(xy00);
            let (elevation10, normal10) = get_one(xy10);
            let (elevation01, normal01) = get_one(xy01);
            let (elevation11, normal11) = get_one(xy11);

            let elevation = math::bilinear_interpolation(elevation00, elevation10, elevation01, elevation11, t);
            let normal: Vec3<f32> = math::bilinear_interpolation::<Vec3<f32>>(normal00.into(), normal10.into(), normal01.into(), normal11.into(), t);

            (elevation, normal.into_array())
        };

        return bilinear_sample(region_uv);

    }

    pub fn from_plot_data_grid(plot_grid: &utils::Grid<PlotTerrainData>, coord: (u32, u32), segment_num: u32, planet_desc: &PlanetDescriptor, axis: math::AxisNormal) -> Self{
        

        let triangles = mesh::Triangles::create_grid_on_unit_cube(
            coord.0, 
            coord.1, 
            axis, 
            segment_num, 
            planet_desc.mesh_grid_segment_num
        );

        let chunk_side_point_num = planet_desc.mesh_grid_segment_num + 1;
    

        let mut vertices = Vec::with_capacity(triangles.0.len());

        for yi in 0..chunk_side_point_num{
            for xi in 0..chunk_side_point_num{
                let index = xi + yi * chunk_side_point_num;
                let position = triangles.0[index as usize];
                let chunk_uv = Vec2::<f32>::new(xi as f32 / chunk_side_point_num as f32, yi as f32 / chunk_side_point_num as f32);
                
                let xy = axis.get_xy(&position);
                let region_uv:Vec2<f32> = xy.map(|s| (s + 1.0)/2.0).into();
                
                let (elevation, normal) = Self::sample_on_grid_plot(plot_grid, region_uv);
                
                let position = Vec3::<f32>::from(position).normalized() * (planet_desc.radius + elevation);
                
                let v: mesh::MeshVertex = mesh::MeshVertex{
                    pos: position.into_array(),
                    normal,
                    tex_coord:chunk_uv.into_array(),
                };

                vertices.push(v);
            }
        };


        Self { mesh: mesh::Mesh { vertices, indices:triangles.1 } }

    
    }

    #[allow(dead_code)]
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

        let vertices = triangles.0.iter().enumerate().map(|(idx, v)|{

            let position: Vec3<f32> = (*v).into();
            let xy = plot_info.axis_normal.get_xy(v);
            let uv = Vec2::new((xy[0] + 1.0) / segment_num as f32, (xy[1] + 1.0) / segment_num as f32);


            let normal = plot_data.cell_normals[idx];
            let position = position.normalized() * planet_desc.radius + plot_data.cell_elevations[idx];

            

        
        

            mesh::MeshVertex{
                pos: position.into_array(),
                normal: normal,
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

    pub fn byte_size(mesh_segment_num: u32) -> u64{
        let vertices_num = (1 + mesh_segment_num).pow(2) as usize;
        let indices_num = mesh_segment_num.pow(2) * 6;
        let vertices_size = vertices_num * std::mem::size_of::<mesh::MeshVertex>();
        let indices_size = (indices_num as usize) * std::mem::size_of::<u32>();
        (vertices_size + indices_size) as u64
    }

    pub fn from_reader<R>(reader: &mut R, mesh_segment_num: u32) -> Result<Self, std::io::Error> where R: std::io::Read{
        let vertices_num = (1 + mesh_segment_num).pow(2) as usize;
        let mut vertices = vec![mesh::MeshVertex::default(); vertices_num];
        let mut indices = vec![0u32; (mesh_segment_num.pow(2) *  6) as usize];

        
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
    pub need_render_chunk: Vec<usize>,

    chunk_datas_reader: std::io::BufReader<std::fs::File>,
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
    const POOL_MAX:u32 = 200;

    pub fn build(axis: math::AxisNormal,region_share:& RegionsShareInfo, planet_desc: &PlanetDescriptor, noise_func: impl Fn(&[f32; 3]) -> (f32, [f32; 3]),planet_dir: std::path::PathBuf) {

        
        //creat a plot_data file
        let plot_data_file_path = planet_dir.join(format!("plot_datas_{}.bin", axis));
        let mut plot_data_file = std::fs::File::create(plot_data_file_path).unwrap();
        let mut plot_data_writer = std::io::BufWriter::new(&mut plot_data_file);
        
        //create a chunk terrain data file
        let chunk_data_file_path = planet_dir.join(format!("chunk_datas_{}.bin", axis));
        let mut chunk_data_file = std::fs::File::create(chunk_data_file_path).unwrap();
        let mut chunk_data_writer = std::io::BufWriter::new(&mut chunk_data_file);

        let mut plot_positions:Vec<[f32; 3]> = vec![[0.0,0.0,0.0]; region_share.region_plots_num as usize];

       
        let mut plot_data_arr = Vec::new();
        for yi in 0..region_share.region_side_plots_num {
            for xi in 0..region_share.region_side_plots_num{
                let plot_index = yi * region_share.region_side_plots_num + xi;
                
                let pos = axis.mat3() * Vec3::new(
                    -1.0 + (xi as f32 + 0.5) * 2.0 / region_share.region_side_plots_num as f32,
                    -1.0 + (yi as f32 + 0.5) * 2.0 / region_share.region_side_plots_num as f32,
                    1.0
                );
                
                //cobe wrap pos
                let mut pos = pos.into_array();
                math::cobe_wrap_with_axis(&mut pos, axis);
                let pos = Vec3::<f32>::from(pos).normalized() * planet_desc.radius;

                

                let cell_positions = mesh::Triangles::create_grid_points_on_unit_cube(xi, yi, axis, region_share.region_side_plots_num, planet_desc.mesh_grid_segment_num).into_iter().map(|p| {
                    let p = Vec3::<f32>::from(p);
                    let p = p.normalized() * planet_desc.radius;
                    p.into_array()
                }).collect();
                let cell_positions = utils::Grid::new_square(planet_desc.mesh_grid_segment_num, cell_positions);
                

                let plot_data = PlotTerrainData::new(cell_positions, &noise_func);
                plot_data.to_writer(&mut plot_data_writer).unwrap();
                
                
                plot_data_arr.push(plot_data);
                
                

                plot_positions[plot_index as usize] = pos.into_array();

            };

            
        };

        let plot_data_grid = utils::Grid::new_square(region_share.region_side_plots_num, plot_data_arr);
        
        
        let mut level_chunk_side_num = 1u16;
        for li in 0..(planet_desc.lod_level + 1) {
            
            for yi in 0..level_chunk_side_num{
                for xi in 0..level_chunk_side_num{

                    let chunk_terrain_data = ChunkTerrainData::from_plot_data_grid(&plot_data_grid, (xi as u32, yi as u32), level_chunk_side_num as u32, &planet_desc, axis);
                    let index = utils::interleave_bit(xi as u16, yi as u16) + utils::LEVELS[li as usize];

                    use std::io::Seek;
                    chunk_data_writer.seek(SeekFrom::Start(index as u64 * ChunkTerrainData::byte_size(planet_desc.mesh_grid_segment_num))).unwrap();

                    chunk_terrain_data.to_writer(&mut chunk_data_writer).unwrap();

                    
                }
            }

            level_chunk_side_num *= 2;
        }

        
         //create a region ron file
        let region_ron_file_path = planet_dir.join(format!("region_{}.ron", axis));
        let mut region_ron_file = std::fs::File::create(region_ron_file_path).unwrap();
        let mut region_ron_writer = std::io::BufWriter::new(&mut region_ron_file);

        let region_ron = RegionRon{
            axis,
            plot_positions_grid: utils::Grid::new_square(region_share.region_side_plots_num, plot_positions),
        };

        ron::ser::to_writer_pretty(&mut region_ron_writer, &region_ron, ron::ser::PrettyConfig::default()).unwrap();
        
    }

    #[allow(dead_code)]
    pub fn load_chunk_datas_from_file(chunk_data_file: fs::File, mesh_segment_num: u32) -> Vec<ChunkTerrainData>{
        let mut chunk_data_reader = std::io::BufReader::new(chunk_data_file);
        let mut chunk_datas = Vec::new();
        loop{
            match ChunkTerrainData::from_reader(&mut chunk_data_reader, mesh_segment_num){
                Ok(chunk_data) => chunk_datas.push(chunk_data),
                Err(_) => break,
            }
        }
        chunk_datas
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

        let chunk_datas_file = std::fs::File::open(planet_dir.join(format!("chunk_datas_{}.bin", axis)))?;

        let chunk_datas_reader = std::io::BufReader::new(chunk_datas_file);
        //return self

        Ok(Self{
            axis,
            plot_positions_grid: region_ron.plot_positions_grid,
            plot_details_grid: plot_details,
            chunk_details,
            plot_datas_grid: utils::Grid::new_square(region_share.region_side_plots_num as u32, plot_datas),
            terrain_chunk_cache,
            need_render_chunk: Vec::new(),
            chunk_datas_reader,
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

    

    pub fn update(&mut self, camera: &camera::Camera, planet_dir: &std::path::PathBuf, mesh_segment_num: u32){

        for i in 0..self.plot_positions_grid.len(){
            let pos:Vec3<f32> =  self.plot_positions_grid[i].into();
            let detail_value = Self::calc_detail_value(camera, &pos, &pos.normalized());
            let coord = self.plot_positions_grid.get_coords(i);
            let zi = utils::interleave_bit(coord.x as u16, coord.y as u16);
            self.plot_details_grid[zi as usize] = detail_value;
        }
        let chunk_len = self.chunk_details.len();
        let plot_len = self.plot_details_grid.len();

  
   
        for i in 0..plot_len {
            self.chunk_details[chunk_len - plot_len + i] = self.plot_details_grid[i];
        
        }

     

        for i in (0..chunk_len-plot_len).rev(){
           
            let sum = self.chunk_details[i*4 +1] + self.chunk_details[i*4 +2] + self.chunk_details[i*4 +3] + self.chunk_details[i*4 +4];
            self.chunk_details[i] = sum ;
        }

        

        self.update_terrain(mesh_segment_num);
    }

    fn update_terrain(&mut self, mesh_segment_num: u32){
        //quad tree dfs chunk details
        let mut stack = Vec::new();
        stack.push(0);

        let mut need_render_chunks = Vec::new();

        while let Some(index) = stack.pop() {
            if self.chunk_details[index] <= 1.0{
                need_render_chunks.push(index);
                continue;
            }

            if index * 4 + 4 < self.chunk_details.len(){
                let children: Vec<usize> = vec![index*4 +1, index*4 +2, index*4 +3, index*4 +4];
                
                for child in children{
                    stack.push(child);
                }
            }else{
                //format panic index not pushed to stack
                panic!("index not pushed to stack");
                
                // need_render_chunks.push(index);
            }
        }

        

        for &index in need_render_chunks.iter(){
            use std::io::Seek;
            if let Some(_) = self.terrain_chunk_cache.get(index as u32){
                self.terrain_chunk_cache.just_put(index as u32);
            } else{

                self.chunk_datas_reader.seek(SeekFrom::Start(index as u64 * ChunkTerrainData::byte_size(mesh_segment_num))).unwrap();
                let terrain_data: ChunkTerrainData = ChunkTerrainData::from_reader(&mut self.chunk_datas_reader, mesh_segment_num).unwrap();
                
                self.terrain_chunk_cache.put(index as u32, terrain_data);
            }
        }

        
        self.need_render_chunk = need_render_chunks;

        if self.need_render_chunk.len() > Self::POOL_MAX as usize{
            panic!("POOL_MAX need to be larger than {}", self.need_render_chunk.len() );
        }
    }




}

pub struct RegionState{

    
    pub terrain_chunk_states_cache: utils::ArrayPool<ChunkTerrainState>,
}

impl RegionState{
    pub fn from_region(_region: &Region, _gpu_agent: &gpu::GpuAgent, region_share: &RegionsShareInfo) -> Self{
        let terrain_chunk_states_cache = utils::ArrayPool::new(Region::POOL_MAX, region_share.region_chunks_num as usize);
        Self{
            terrain_chunk_states_cache,
        }
    }

    pub fn update(&mut self, region: &Region, gpu_agent: &gpu::GpuAgent, _region_share: &RegionsShareInfo){

        for &i in region.need_render_chunk.iter(){
            let chunk = region.terrain_chunk_cache.get(i as u32).unwrap_or_else(|| panic!("chunk {} not loaded", i) );
            let chunk_state = ChunkTerrainState::from_chunk(chunk, gpu_agent);
            self.terrain_chunk_states_cache.put(i as u32, chunk_state);
        }

     
    }

    pub fn get_visible_terrain_states(&self, region: &Region, _region_share: &RegionsShareInfo) -> Vec<(&ChunkTerrainState, u32)>{
        let mut visible = Vec::new();
        for &i in region.need_render_chunk.iter() {
            let chunk_state = self.terrain_chunk_states_cache.get(i as u32).unwrap_or_else(|| panic!("chunk state {} not loaded", i) );
            visible.push((chunk_state, i as u32));
        }

      
        
        visible
    }


}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SunLight{
    pub direction: [f32; 4],
    pub color: [f32; 4],
}

pub struct PlanetState{
    
    terrain_render_pipeline: wgpu::RenderPipeline,
    // sea_render_pipeline: wgpu::RenderPipeline,
    // atomsphere_render_pipeline: wgpu::RenderPipeline,
    region_states: [RegionState; 6],
    instance_buffer: wgpu::Buffer,
    _sun_light_uniform_buffer: wgpu::Buffer,
    sun_light_bind_group: wgpu::BindGroup,
    
    debug_color_uniform_buffer: wgpu::Buffer,
    debug_color_bind_group: wgpu::BindGroup,
}

impl PlanetState {
    

    pub fn new(agent: & gpu::GpuAgent, planet: & Planet) -> Self{
        let transfrom_instance = mesh::TransformInstance{
            mat4: planet.calc_transform_mat4().into_col_arrays(),
        };

        let instance_buffer = agent.create_buffer(
            &[transfrom_instance],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            "planet instance buffer"
        );

        let sun_light = SunLight{
            direction: [0.0, 0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
        };

        let sun_light_uniform_buffer = agent.create_buffer(
            &[sun_light],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "planet sun light uniform buffer"
        );

        let debug_color_uniform_buffer = agent.create_buffer(
            &[[1.0, 0.0, 0.0 ,1.0]],
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            "planet debug color uniform buffer"
        );

        let sun_light_bind_group_layout = agent.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            entries: &[wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("planet sun light bind group layout")
        });

        let sun_light_bind_group = agent.device.create_bind_group(&wgpu::BindGroupDescriptor{
            layout: &sun_light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry{
                binding: 0,
                resource: sun_light_uniform_buffer.as_entire_binding(),
            }],
            label: Some("planet sun light bind group")
        });

        let debug_color_bind_group_layout = agent.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            entries: &[wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("planet debug color bind group layout")
        });

        let debug_color_bind_group = agent.device.create_bind_group(&wgpu::BindGroupDescriptor{
            layout: &debug_color_bind_group_layout,
            entries: &[wgpu::BindGroupEntry{
                binding: 0,
                resource: debug_color_uniform_buffer.as_entire_binding(),
            }],
            label: Some("planet debug color bind group")
        });

        use mesh::VertexLayout;

        let color_push_constant_range = wgpu::PushConstantRange{
            stages: wgpu::ShaderStages::FRAGMENT,
            range: camera::Camera::size_of_push_constant()..camera::Camera::size_of_push_constant()+4,
        };
        
        let terrain_pipeline_layout = agent.create_pipeline_layout(&[&sun_light_bind_group_layout], &[camera::Camera::PUSH_CONSTANT_RANGE], "terrain pipeline layout");

        let terrain_shader = agent.create_shader_from_path("src/shaders/terrain_normal.wgsl");

        let terrain_render_pipeline = agent.create_render_pipeline(
            &terrain_pipeline_layout, 
            &[mesh::MeshVertex::vertex_layout::<0>(), mesh::TransformInstance::vertex_layout::<5>()],
            &terrain_shader, 
            wgpu::PolygonMode::Line,
            agent.config.format, 
            texture::Texture::DEPTH_FORMAT, 
            Some(wgpu::Face::Back),
            "terrain render pipeline"
        );

        // let sea_pipeline_layout = agent.create_pipeline_layout(&[], &[camera::Camera::PUSH_CONSTANT_RANGE], "sea pipeline layout");

        // let sea_shader = agent.create_shader_from_path("src/shaders/sea.wgsl");
        // let sea_render_pipeline = agent.create_render_pipeline(
        //     &sea_pipeline_layout,
        //     &[], 
        //     &sea_shader, 
        //     wgpu::PolygonMode::Fill, 
        //     agent.config.format, 
        //     texture::Texture::DEPTH_FORMAT, 
        //     Some(wgpu::Face::Back),
        //     "sea render pipeline"
        // );

        // let atomsphere_pipeline_layout = agent.create_pipeline_layout(&[], &[camera::Camera::PUSH_CONSTANT_RANGE], "atomsphere pipeline layout");

        // let atomsphere_shader = agent.create_shader_from_path("src/shaders/atomsphere.wgsl");
        // let atomsphere_render_pipeline = agent.create_render_pipeline(
        //     &atomsphere_pipeline_layout, 
        //     &[], 
        //     &atomsphere_shader, 
        //     wgpu::PolygonMode::Fill, 
        //     agent.config.format, 
        //     texture::Texture::DEPTH_FORMAT, 
        //     Some(wgpu::Face::Back),
        //     "atomsphere render pipeline"
        // );

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
            // sea_render_pipeline,
            // atomsphere_render_pipeline,
         
            region_states,
            instance_buffer,
            sun_light_bind_group,
            _sun_light_uniform_buffer: sun_light_uniform_buffer,

            debug_color_bind_group,
            debug_color_uniform_buffer,
        }
    }

    pub fn change_debug_color(&mut self, agent: &gpu::GpuAgent, color: [f32; 4]){
        agent.queue.write_buffer(&self.debug_color_uniform_buffer, 0, bytemuck::cast_slice(&[color]));
    }
    

    pub fn update(&mut self,agent: &gpu::GpuAgent, planet: & Planet){
        
        for ri in 0..6{
            self.region_states[ri].update(&planet.regions[ri], agent, &planet.regions_share);
        }

    }

    pub fn render_terrain<'a>(
        &'a self, 
        render_pass: & mut wgpu::RenderPass<'a>, 
        camera: & camera::Camera,
        planet: & Planet,
        level_filter: impl Fn (u32) -> bool,
    ) {
        render_pass.set_pipeline(&self.terrain_render_pipeline);
        render_pass.set_bind_group(0, &self.sun_light_bind_group, &[]);
        render_pass.set_push_constants(wgpu::ShaderStages::VERTEX_FRAGMENT, 0, camera.get_uniform_data() );
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        
        for ri in 0..6{
            
            let terrain_states = self.region_states[ri].get_visible_terrain_states(&planet.regions[ri], &planet.regions_share);
            
            for ( chunk_state, index) in terrain_states.iter() {
                let level = utils::get_quad_tree_level(*index);
               
           
                render_pass.set_vertex_buffer(0, chunk_state.mesh_state.vertex_buffer.slice(..));
                render_pass.set_index_buffer(chunk_state.mesh_state.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk_state.mesh_state.index_count, 0, 0..1);
            }
        }
    }

    
}











pub struct Planet{
    pub planet_dir: std::path::PathBuf,
    pub name: Name,
    pub radius: f32,
    pub lod_level: u8,
    pub position: Vec3<f32>,
    pub rotation: Quaternion<f32>,

    pub regions: [Region; 6],
    pub regions_share: RegionsShareInfo,

    pub plots_side_num: u32,
    pub mesh_grid_segment_num: u32,
    pub terrain_elevation_bounds: (f32, f32),
    pub seed: u32,
}

#[derive(Serialize, Deserialize)]
pub struct PlanetDescriptor{
    pub name: Name,
    pub radius: f32,
    pub lod_level: u8,
    pub position: Vec3<f32>,
    pub rotation: Quaternion<f32>,
    pub mesh_grid_segment_num: u32,
    pub seed: u32,
    pub terrain_elevation_bounds: (f32, f32),

}

impl PlanetDescriptor{
    pub fn calc_plots_side_num(&self) -> u32{
        2u32.pow(self.lod_level as u32)
    }

    
}





impl Planet{
    

    
    
    pub fn build(planet_desc: &PlanetDescriptor, planet_dir: std::path::PathBuf) -> Result<(), std::io::Error>{
        
        {//descriptor ron file
            let ron_file_path = planet_dir.join("planet").with_extension("ron");
            //if not exist, create it
       
            let ron_file = std::fs::File::create(ron_file_path)?;
            let mut ron_writer = std::io::BufWriter::new(ron_file);
            ron::ser::to_writer_pretty(&mut ron_writer, &planet_desc, ron::ser::PrettyConfig::default()).unwrap();
        }

        
        

        {//build regions
            use noise::{NoiseFn, Perlin, Fbm};

           
            let fbm = Fbm::<Perlin>::new(0);
            let noise_func = |pos: &[f32; 3]| ->(f32, [f32; 3]) {
                let pos_f64 = pos.map(|s| (s/planet_desc.radius) as f64);
                let elevation = utils::map01_to_bound( fbm.get(pos_f64) as f32, planet_desc.terrain_elevation_bounds);

                
                
                let pos = Vec3::<f32>::from(*pos);
                let normal = pos.normalized().into_array();
                (elevation, normal)
            };
            let info = RegionsShareInfo::new(planet_desc.lod_level);
            // let plot_side_num = planet_desc.calc_plots_side_num();
            for ri in 0..6{
                let axis = math::AxisNormal::from_u32(ri);
                Region::build(axis, &info, planet_desc, noise_func, planet_dir.clone());

                
            }
        };
        
        Ok(())
    }

    pub fn load_from_dir(planet_dir: std::path::PathBuf, planet_name: utils::Name) -> Result<Self, std::io::Error>{
        //load descriptor from  exsited ron file
        

        let planet_desc: PlanetDescriptor = {
            let ron_file_path = planet_dir.join("planet").with_extension("ron");
            let ron_file = fs::File::open(ron_file_path).unwrap();
            let mut ron_reader = std::io::BufReader::new(ron_file);
            ron::de::from_reader(&mut ron_reader).unwrap()
        };

        let region_share = RegionsShareInfo::new(planet_desc.lod_level);

        
        //load regions
        let regions = math::AxisNormal::AXIS_ARRAY.map(|axis| {
            Region::load_from_dir(axis, planet_dir.clone(), &region_share, &planet_desc).unwrap()
        });

        let plot_side_num = planet_desc.calc_plots_side_num();

        Ok(Self{
            planet_dir,
            name: planet_name,
            radius: planet_desc.radius,
            lod_level: planet_desc.lod_level,
            position: planet_desc.position,
            rotation: planet_desc.rotation,
            regions: regions,
            regions_share: region_share,
            plots_side_num: plot_side_num,
            mesh_grid_segment_num: planet_desc.mesh_grid_segment_num,
            terrain_elevation_bounds: planet_desc.terrain_elevation_bounds,
            seed: planet_desc.seed,
        })

    }
    
    #[allow(dead_code)]
    pub fn description(&self) ->PlanetDescriptor {
        PlanetDescriptor{
            name: self.name.clone(),
            position: self.position,
            radius: self.radius,
            lod_level: self.lod_level,
            rotation: self.rotation,
            mesh_grid_segment_num: self.mesh_grid_segment_num,
            seed: self.seed,
            terrain_elevation_bounds: self.terrain_elevation_bounds,
        }
    }

    pub fn calc_transform_mat4(&self) -> Mat4<f32> {
        //construct transform matrix with position and rotation
        Mat4::from(Transform{
            position: self.position,
            orientation: self.rotation,
            scale: Vec3::one(),
        })
        
    }

    pub fn update(&mut self, _delta_time: f32, camera: &camera::Camera){
        for i in 0..6{
            self.regions[i].update(camera, &self.planet_dir ,self.mesh_grid_segment_num);
            
        }
    }

 

    
 
  
}




//test
#[cfg(test)]
mod test{
    use super::*;
    use crate::utils;
    use crate::mesh;


    #[test]
    fn test_planet_build(){
        //enable backtrace
        std::env::set_var("RUST_BACKTRACE", "1");

        let planet_desc = PlanetDescriptor{
            name: utils::Name::new("AA"),
            radius: 100.0,
            lod_level: 4,
            position: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quaternion::identity(),
            mesh_grid_segment_num: 16,
            seed: 0,
            terrain_elevation_bounds: (-3.0, 3.0),
        };
        let planet_dir = std::path::PathBuf::from("test/planet/AA");
        //create planet dir if not exsited
        if !planet_dir.exists(){
            fs::create_dir_all(&planet_dir).unwrap();
        }

        
        Planet::build(&planet_desc, planet_dir.clone()).unwrap();

        //load planet
        let _planet: Planet = Planet::load_from_dir(planet_dir.clone(), planet_desc.name).unwrap();
    }

    #[test]
    fn test_chunk_data(){
        let path = std::path::PathBuf::from("test/planet/AA/chunk_datas_mX.bin");
        let file = std::fs::File::open(path).unwrap();
        let chunk_datas = Region::load_chunk_datas_from_file(file, 16);

        let mut mesh = mesh::Mesh::empty();

        //merge mesh
        for chunk_data in chunk_datas.iter(){
            mesh.merge(&chunk_data.mesh);
        }
        
        let dir_path = std::path::PathBuf::from("test/planet/AA/objs");
        if !dir_path.exists(){
            fs::create_dir_all(dir_path.clone()).unwrap();
        }

        let mut index = 0;
        for li in 0..5{
            let level_num = 4u32.pow(li);
            let mut mesh = mesh::Mesh::empty();
            for _ in 0..level_num{
                mesh.merge(&chunk_datas[index].mesh);
                index += 1;
            }
            let path = dir_path.clone().join(format!("mesh_level_{}.obj", li));
            mesh.save_obj_file(path);

            
        }
        
        
        //save mesh
        let path = dir_path.clone().join("mesh.obj");
        mesh.save_obj_file(path);
    
        for (idx, chunk_data) in chunk_datas.iter().enumerate(){
            let path = dir_path.clone().join(format!("chunk_data_{}.obj", idx));
     
            chunk_data.mesh.save_obj_file(path);
        }
    }

}



