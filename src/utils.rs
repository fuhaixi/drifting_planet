use std::ops::{Index, IndexMut};
use std::path::Path;
use std::collections::VecDeque;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Name{
    inner: [u8; 32],
}

//impl serialize for name
impl serde::Serialize for Name{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer{
        
        serializer.serialize_str(self.as_str())
    }
}

//impl deserialize for name
impl<'de> serde::Deserialize<'de> for Name{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: serde::Deserializer<'de>{
        let s = String::deserialize(deserializer)?;
        Ok(Name::new(&s))
    }
}


impl Name{
    pub fn new(name: &str) -> Self{
        let mut inner = [0; 32];
        inner[..name.len()].copy_from_slice(name.as_bytes());
        Self{inner}
    }

    #[allow(dead_code)]
    pub fn to_string(&self) -> String{
        self.as_str().to_string()
    }

    #[allow(dead_code)]
    pub fn as_bytes(&self) -> &[u8]{
        &self.inner
    }

    #[allow(dead_code)]
    pub fn as_str(&self) -> &str{
        //remove trailing zeros
        let mut len = 0;
        for i in 0..self.inner.len(){
            if self.inner[i] != 0{
                len = i+1;
            }
        }
        std::str::from_utf8(&self.inner[..len]).unwrap()
    }
}


//impl dsiplay for name
impl std::fmt::Display for Name{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        write!(f, "{}", self.as_str())
    }
}


use vek::*;
#[derive(serde::Serialize, serde::Deserialize) ]
pub struct Grid<T>{
    pub extent: Extent2<u32>,
    pub data: Vec<T>,
}

impl<T> Grid<T> where T: Clone{
    pub fn new_with_default(extent: Extent2<u32>, default: T) -> Self{
        let data = std::iter::repeat(default).take((extent.w * extent.h) as usize).collect::<Vec<_>>();
        Self{extent, data}
    }

    pub fn new_square_with_default(side_num: u32, default: T) -> Self{
        Self::new_with_default(Extent2::new(side_num, side_num), default)
    }
}

impl<T> Grid<T>{
    #[allow(dead_code)]
    pub fn new(extent: Extent2<u32>, data: Vec<T>) -> Self{
        Self{extent, data}
    }

   

    #[allow(dead_code)]
    pub fn new_square(side_num: u32, data: Vec<T>) -> Self{
        Self::new(Extent2::new(side_num, side_num), data)
    }

   

    #[allow(dead_code)]
    pub fn get(&self, pos: Vec2<u32>) -> Option<&T>{
        let index = self.get_index(pos);
        self.data.get(index)
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, pos: Vec2<u32>) -> Option<&mut T>{
        let index = self.get_index(pos);
        self.data.get_mut(index)
    }

    #[allow(dead_code)]
    pub fn get_index(&self, pos: Vec2<u32>) -> usize{
        let x = pos.x ;
        let y = pos.y;

        (y * self.extent.w + x) as usize
    }

    pub fn get_coords(&self, index: usize) -> Vec2<u32>{
        let x = index as u32 % self.extent.w;
        let y = index as u32 / self.extent.w;
        Vec2::new(x, y)
    }

    #[allow(dead_code)]
    pub fn set(&mut self, pos: Vec2<u32>, value: T){
        let index = self.get_index(pos);
        self.data[index] = value;
    }

    #[allow(dead_code)]
    pub fn map<F, P: Clone>(&self, f: F) -> Grid<P> where F: Fn(&T) -> P{
        let data = self.data.iter().map(f).collect::<Vec<_>>();
        Grid::new(self.extent, data)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize{
        self.data.len()
    }

    #[allow(dead_code)]
    pub fn iter(&self) -> std::slice::Iter<T>{
        self.data.iter()
    }
}

pub const LEVELS:[u32; 17] = [0, 1, 5, 21, 85, 341, 1365, 5461, 21845, 87381, 349525, 1398101, 5592405, 22369621, 89478485, 357913941, 1431655765];
pub fn get_quad_tree_level(index: u32) -> u32 {
    let mut level = 0;
    for i in 1..LEVELS.len(){
        if index < LEVELS[i]{
            level = (i-1) as u32;
            break;
        }
    }
    level
}

impl Grid<f32>{
    
    #[allow(dead_code)]
    pub fn bilinear_sample(&self, uv: Vec2<f32>) -> f32 {
        let x = uv.x * self.extent.w as f32;
        let y = uv.y * self.extent.h as f32;

        let x0 = x.floor() as u32;
        let x1 = x.ceil() as u32;
        let y0 = y.floor() as u32;
        let y1 = y.ceil() as u32;

        let x0 = x0.min(self.extent.w - 1);
        let x1 = x1.min(self.extent.w - 1);
        let y0 = y0.min(self.extent.h - 1);
        let y1 = y1.min(self.extent.h - 1);

        let q00 = self.get(Vec2::new(x0, y0)).unwrap();
        let q01 = self.get(Vec2::new(x0, y1)).unwrap();
        let q10 = self.get(Vec2::new(x1, y0)).unwrap();
        let q11 = self.get(Vec2::new(x1, y1)).unwrap();

        let s = x - x0 as f32;
        let t = y - y0 as f32;

        let q0 = (*q10 )* s + (*q00) * (1.0 - s) ;
        let q1 = (*q11) * s + (*q01) * (1.0 - s) ;

        let ret = q1 * t + q0 * (1.0 - t);
        ret
    }

  
    
}

//implement iter() for grid
impl<T> IntoIterator for Grid<T>{
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter{
        self.data.into_iter()
    }
}



impl<T> Index<usize> for Grid<T>{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output{
        &self.data[index]
    }
}

//index mut
impl<T> IndexMut<usize> for Grid<T>{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output{
        &mut self.data[index]
    }
}




pub struct ArrayPool<T>{
    pool: Vec<Option<(Box<T>, u32)>>,
    queue: VecDeque<u32>,//index of pool
    max: u32,
}

impl <T> ArrayPool<T> {
    pub fn new(max: u32, array_capacity: usize) -> Self{
        Self{
            pool: std::iter::repeat_with(|| None).take(array_capacity).collect::<Vec<_>>(),
            queue:VecDeque::with_capacity(max as usize),
            max,
        }
    }

    pub fn get(&self, index: u32) -> Option<&T>{

        self.pool[index as usize].as_ref().map(|x| {
                
            x.0.as_ref()}
        )
    }

    pub fn just_put(&mut self, index: u32){
        if self.queue.len() < self.max as usize{
            self.queue.push_back(index);
        }
        else{
            self.queue.push_back(index);
            let old_index = self.queue.pop_front().unwrap();
            if self.pool[old_index as usize].as_ref().unwrap().1 == 1{
                if old_index == index {
                    self.pool[old_index as usize].as_mut().unwrap().1 -= 1;
                }
                else{

                    self.pool[old_index as usize] = None;
                }
            }
            else{
                self.pool[old_index as usize].as_mut().unwrap().1 -= 1;
            }
        }
        self.pool[index as usize].as_mut().unwrap().1 += 1;
    }

    #[allow(dead_code)]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T>{
        self.pool[index as usize].as_mut().map(|x| x.0.as_mut())
    }

    #[allow(dead_code)]
    pub fn put(&mut self, index: u32, value: T) -> u32{
        assert!(index < self.pool.len() as u32);

        if self.queue.len() < self.max as usize{
            if let Some(a) = self.pool[index as usize].as_mut(){
                a.0 = Box::new(value);
                a.1 += 1;
                self.queue.push_back(index);

                return a.1;
            }
            else{
                self.pool[index as usize] = Some((Box::new(value), 1));
                self.queue.push_back(index);
                return 1;
            }
        }else{
            let old_index = self.queue.pop_front().unwrap();
            
            if let Some(a) = &self.pool[old_index as usize]{
                if a.1 == 1{
                    self.pool[old_index as usize] = None;
                }
                else{
                    self.pool[old_index as usize].as_mut().unwrap().1 -= 1;
                }
            }

            
            if let Some(a) = self.pool[index as usize].as_mut(){
                a.0 = Box::new(value);
                a.1 += 1;
                self.queue.push_back(index);

                return a.1;
            }
            else{
                self.pool[index as usize] = Some((Box::new(value), 1));
                self.queue.push_back(index);
                return 1;
            }
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize{
        self.queue.len()
    }

    
}


impl AsRef<Path> for Name{
    fn as_ref(&self) -> &Path{
        Path::new(self.as_str())
    }
}

#[allow(dead_code)]
pub fn create_new_file<P>(path: P) -> Result<std::fs::File, std::io::Error> where P: AsRef<Path>{
    if !path.as_ref().exists() {
        std::fs::File::create(path)
    }
    else{
        Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "file already exists"))
    }
}



#[allow(dead_code)]
pub fn map01_to_bound(value01: f32, bound: (f32, f32) ) -> f32{
    value01 * (bound.1 - bound.0) + bound.0
}

#[allow(dead_code)]
pub fn interleave_bit(x: u16, y: u16) -> u32{
    let mut z = 0u32;
    let x = x as u32;
    let y = y as u32;
    for i in 0..16{
        z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    z
}

#[allow(dead_code)]
pub fn de_interleave_bit(z: u32) -> (u16, u16){
    let mut x = 0u16;
    let mut y = 0u16;
    for i in 0..16{
        x |= ((z & (1 << (2 * i))) >> i) as u16;
        y |= ((z & (1 << (2 * i + 1))) >> (i + 1)) as u16;
    }
    (x, y)
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct NoiseDescriptor {
    pub seed: u32,
    pub octaves: usize,
    pub frequency: f64,
    pub lacunarity: f64,
    pub persistence: f64,
}

use noise::{Fbm, MultiFractal, Seedable, OpenSimplex};

pub fn build_noise<T>(descriptor: &NoiseDescriptor) -> Fbm<T>
where
    T: Default + Seedable,
{
    Fbm::new(descriptor.seed)
        .set_octaves(descriptor.octaves)
        .set_frequency(descriptor.frequency)
        .set_lacunarity(descriptor.lacunarity)
        .set_persistence(descriptor.persistence)
}

pub fn build_open_simplex_noise(descriptor: &NoiseDescriptor) -> Fbm<OpenSimplex> {
    build_noise(descriptor)
}

#[cfg(test)]
mod test{
    use super::*;
    
    #[test]
    fn test_name(){
        let name = Name::new("test");
        assert_eq!(name.as_str(), "test");
        assert_eq!(name.as_bytes(), b"test");
        assert_eq!(name.to_string(), "test");
    }


    #[test]
    fn test_grid(){
        let grid = Grid::new(Extent2::new(2, 2), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(grid.get(Vec2::new(0, 0)), Some(&1.0));
        assert_eq!(grid.get(Vec2::new(1, 0)), Some(&2.0));
        assert_eq!(grid.get(Vec2::new(0, 1)), Some(&3.0));
        assert_eq!(grid.get(Vec2::new(1, 1)), Some(&4.0));
        assert_eq!(grid.get(Vec2::new(2, 1)), None);
        assert_eq!(grid.get(Vec2::new(1, 2)), None);
        assert_eq!(grid.get(Vec2::new(2, 2)), None);

        for (i, v) in grid.iter().enumerate(){
            assert_eq!(*v, (i + 1) as f32);
        }
        
    }

    //test serilize and deserilize name to ron
    #[test]
    fn test_serde_name(){
        let name = Name::new("test");
        let ron = ron::ser::to_string(&name).unwrap();
        println!("{}", ron);
        let name2: Name = ron::de::from_str(&ron).unwrap();
        assert_eq!(name, name2);
    }

    //test array pool
    #[test]
    fn test_array_pool(){
        let mut pool = ArrayPool::<String>::new(4, 10);

        assert_eq!(pool.put(0, "0".to_string()), 1);
        pool.put(0, "0".to_string());
        pool.put(1, "1".to_string());
        pool.put(2, "2".to_string());
        pool.put(3, "3".to_string());

        assert_eq!(pool.put(4, "4".to_string()), 1);
        assert_eq!(pool.put(5, "5".to_string()), 1);
        pool.put(0, "1".to_string());

        assert_eq!(pool.get(0), Some(& "1".to_string()) );

        //put 1 again
        assert_eq!(pool.put(0, "1".to_string()), 2);
        assert_eq!(pool.get(0), Some(& "1".to_string()) );

        //put 2 again
        assert_eq!(pool.put(2, "2.2".to_string()), 1);
        assert_eq!(pool.get(2), Some(& "2.2".to_string()) );
    }

    #[test]
    fn test_interleave_bit(){
        assert_eq!(interleave_bit(1, 3), 11);
        assert_eq!(interleave_bit(3, 5), 39);

        //test de_interleave_bit
        assert_eq!(de_interleave_bit(11), (1, 3));
        assert_eq!(de_interleave_bit(39), (3, 5));
    }

}

