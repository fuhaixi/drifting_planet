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

impl<T> Grid<T>{
    pub fn new(extent: Extent2<u32>, data: Vec<T>) -> Self{
        Self{extent, data}
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

    pub fn sample
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

