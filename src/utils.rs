use std::ops::{Index, IndexMut};
use std::path::Path;
use std::collections::VecDeque;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct Name{
    inner: [u8; 32],
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



use vek::*;
pub struct Grid<T>{
    pub extent: Extent2<u32>,
    pub data: Vec<T>,
}

impl<T> Grid<T> where T: Clone{
    pub fn new(extent: Extent2<u32>, data: Vec<T>) -> Self{
        Self{extent, data}
    }

    pub fn new_with_default(extent: Extent2<u32>, default: T) -> Self{
        let data = std::iter::repeat(default).take((extent.w * extent.h) as usize).collect::<Vec<_>>();
        Self{extent, data}
    }

    pub fn new_square(side_num: u32, data: Vec<T>) -> Self{
        Self::new(Extent2::new(side_num, side_num), data)
    }

    pub fn new_square_with_default(side_num: u32, default: T) -> Self{
        Self::new_with_default(Extent2::new(side_num, side_num), default)
    }

    pub fn get(&self, pos: Vec2<u32>) -> Option<&T>{
        let index = self.get_index(pos);
        self.data.get(index)
    }

    pub fn get_mut(&mut self, pos: Vec2<u32>) -> Option<&mut T>{
        let index = self.get_index(pos);
        self.data.get_mut(index)
    }

    pub fn get_index(&self, pos: Vec2<u32>) -> usize{
        let x = pos.x ;
        let y = pos.y;

        (y * self.extent.w + x) as usize
    }

    pub fn set(&mut self, pos: Vec2<u32>, value: T){
        let index = self.get_index(pos);
        self.data[index] = value;
    }

    pub fn map<F, P: Clone>(&self, f: F) -> Grid<P> where F: Fn(&T) -> P{
        let data = self.data.iter().map(f).collect::<Vec<_>>();
        Grid::new(self.extent, data)
    }
}

impl<T> Grid<T> where T: std::ops::Mul<f32, Output = T> + std::ops::Add<T, Output =  T> + Clone{
    
    pub fn bilinear_sample(&self, uv: Vec2<f32>) -> T {
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

    pub fn len(&self) -> usize{
        self.data.len()
    }

    pub fn iter(&self) -> std::slice::Iter<T>{
        self.data.iter()
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
    pool: Vec<Option<Box<T>>>,
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
                
            x.as_ref()}
        )
    }

    pub fn get_mut(&mut self, index: u32) -> Option<&mut T>{
        self.pool[index as usize].as_mut().map(|x| x.as_mut())
    }

    pub fn put(&mut self, index: u32, value: T){
        if self.queue.len() < self.max as usize{
            self.pool[index as usize] = Some(Box::new(value));
            
            self.queue.push_back(index);

        }else{
            let old_index = self.queue.pop_front().unwrap();
            self.pool[old_index as usize] = None;
            self.pool[index as usize] = Some(Box::new(value));
            self.queue.push_back(index);
        }
    }

    pub fn len(&self) -> usize{
        self.queue.len()
    }

    
}


impl AsRef<Path> for Name{
    fn as_ref(&self) -> &Path{
        Path::new(self.as_str())
    }
}

pub fn create_new_file<P>(path: P) -> Result<std::fs::File, std::io::Error> where P: AsRef<Path>{
    if !path.as_ref().exists() {
        std::fs::File::create(path)
    }
    else{
        Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "file already exists"))
    }
}


#[cfg(test)]
mod test{
    use super::*;
    use vek::*;
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
}
