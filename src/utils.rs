use std::path::Path;
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