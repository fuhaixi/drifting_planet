use crate::mesh;

pub struct ChunkMesh{

}

pub struct ChunkData{
    mesh: ChunkMesh
}

pub struct ChunkHead{
    corners: [[f32; 3]; 4],
    raw_corners: [[f32; 2]; 4],
    normal: [f32; 3],
    index: u32,
    depth: u8,
}




