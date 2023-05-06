
#[repr(C)]
#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
pub enum AxisNormal{
    X = 0,
    Y = 1,
    Z = 2,
    Mx = 3,
    My = 4,
    Mz = 5
}
//impl display for AxisNormal
use std::fmt;
impl fmt::Display for AxisNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            AxisNormal::X => write!(f, "X"),
            AxisNormal::Y => write!(f, "Y"),
            AxisNormal::Z => write!(f, "Z"),
            AxisNormal::Mx => write!(f, "mX"),
            AxisNormal::My => write!(f, "mY"),
            AxisNormal::Mz => write!(f, "mZ"),
        }
    }
}


impl AxisNormal {
    pub const AXIS_ARRAY:[AxisNormal; 6] = [AxisNormal::X, AxisNormal::Y, AxisNormal::Z, AxisNormal::Mx, AxisNormal::My, AxisNormal::Mz];

    #[allow(dead_code)]
    pub fn from_u32(num: u32) -> Self{
        let num = num % 6;
        unsafe{
            std::mem::transmute(num)
        }
    }
    /// z
    #[allow(dead_code)]
    pub fn normal(&self) -> [f32; 3] {
        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as i32 ;
        let sig = if n < 3 {1.0} else {-1.0};
        arr[n as usize % 3] = sig;
        arr
    }

    pub fn index(&self) -> usize {
        *self as usize
    }
    
    /// x
    #[allow(dead_code)]
    pub fn tangent(&self) -> [f32; 3] {

        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as i32 ;
        let sig:i32 = if n < 3 {1} else {-1};
        arr[(n + 1*sig + 3) as usize % 3] = sig as f32;
        arr
    }

    /// y
    #[allow(dead_code)]
    pub fn btangent(&self) -> [f32; 3] {

        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as i32 ;
        let sig = if n < 3 {1} else {-1};
        arr[(n + 2*sig + 3) as usize % 3] = sig as f32;
        arr
    }

    pub fn mat3_col_arrays(&self) -> [[f32; 3]; 3] {
        let mut arr : [[f32; 3]; 3] = [[0.0; 3]; 3];
        let n: i32 = *self as i32 ;
        let sig = if n < 3 {1} else {-1};

        arr[0][(n + 1*sig + 3) as usize % 3] = sig as f32;
        arr[1][(n + 2*sig + 3) as usize % 3] = sig as f32;
        arr[2][n  as usize % 3]= sig as f32;
        arr
    }

    pub fn mat3(&self) -> Mat3<f32> {
        Mat3::from_col_arrays(self.mat3_col_arrays())
    }


    pub fn get_xy(&self, vec:&[f32; 3]) -> [f32; 2] {
        let n = *self as i32 ;
        let sig = if n < 3 {1} else {-1};
        [vec[(n + 1*sig + 3) as usize % 3] * sig as f32, vec[(n + 2*sig + 3) as usize % 3] * sig as f32]
    }

    pub fn set_xy(&self, vec:&mut [f32; 3], xy : [f32; 2]) {
        let n = *self as i32 ;
        let sig = if n < 3 {1} else {-1};
        vec[(n + 1*sig + 3) as usize % 3] = xy[0] * sig as f32;
        vec[(n + 2*sig + 3) as usize % 3] = xy[1] * sig as f32;
    }
}

use vek::*;
pub fn cobe_wrap( x: Vec2<f32>) -> Vec2<f32> {
    let y = x.yx();
    let x2 = x*x;
    let y2= y*y;
    let bsum =  (-0.0941180085824 + 0.0409125981187*y2 - 0.0623272690881*x2)*x2 +
                    (0.0275922480902 + 0.0342217026979*y2)*y2 ;
    return ( 0.723951234952 + 0.276048765048*x2 + (Vec2::one() - x2)*bsum )*x;
}

pub fn cobe_wrap_with_axis(vec:&mut [f32; 3], axis: AxisNormal){

    let n = axis as usize ;
    let x = Vec2::new(vec[(n + 1) % 3], vec[(n + 2) % 3]);
    let y = cobe_wrap(x);

    vec[(n + 1) % 3] = y.x;
    vec[(n + 2) % 3] = y.y;
    
}

/// bilinear interpolation
pub fn bilinear_interpolation<T> (g00: T, g01: T, g10: T, g11: T, t: Vec2<f32>) -> T where T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + Copy {
    let a = g00 * (1.0 - t[0]) + g10 * t[0];
    let b = g01 * (1.0 - t[0]) + g11 * t[0];
    a * (1.0 - t[1]) + b * t[1]
}


#[cfg(test)]
mod test{
    use super::*;
    use rand;
    #[test]
    fn test_get_xy(){
        //generate a random vector3
        for _i in 0..100{
            let vec: Vec3<f32> = Vec3::new(rand::random::<f32>(), rand::random::<f32>(), rand::random::<f32>());

            let axis = AxisNormal::from_u32(rand::random::<u32>());
            //get the x and y of the vector3
            let xy = axis.get_xy(&vec.into_array());

            let v = Mat3::<f32>::from_col_arrays(axis.mat3_col_arrays()).transposed() * vec;

            let xy2 = [v.x, v.y];

            
            assert_eq!(xy, xy2);
        }
    }

    #[test]
    fn test_axis_normal(){
        let axis = AxisNormal::X;
        assert_eq!(axis.normal(), [1.0, 0.0, 0.0]);
        assert_eq!(axis.tangent(), [0.0, 1.0, 0.0]);
        assert_eq!(axis.btangent(), [0.0, 0.0, 1.0]);
    }

    use vek::*;
    #[test]
    fn test_axis_normal_mat3(){
        for axis in AxisNormal::AXIS_ARRAY.iter(){
            let mat3 = Mat3::from_col_arrays([ axis.tangent(), axis.btangent() , axis.normal()]);
            let mat3_2 =  Mat3::from_col_arrays(axis.mat3_col_arrays());
            //check if mat3 and mat3_2 is equal
            assert_eq!(mat3, mat3_2);

            //check if mat3 and mat3_2 is orthogonal
            assert_eq!(mat3 * mat3.transposed(), Mat3::identity());
            
            
        }

    }

    #[test]
    fn test_axis(){
        for axis in AxisNormal::AXIS_ARRAY.iter(){
            let tangent = Vec3::<f32>::from(axis.tangent());
            let btangent = Vec3::<f32>::from(axis.btangent());
            let normal = Vec3::<f32>::from(axis.normal());
            assert_eq!(tangent.cross(btangent), normal);
        }
    }

    
}