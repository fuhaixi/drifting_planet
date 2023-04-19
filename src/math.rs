

#[derive(Copy, Clone)]
pub enum AxisNormal{
    X = 0,
    Y = 1,
    Z = 2,
    Mx = 3,
    My = 4,
    Mz = 5
}


impl AxisNormal {
    pub const AXIS_ARRAY:[AxisNormal; 6] = [AxisNormal::X, AxisNormal::Y, AxisNormal::Z, AxisNormal::Mx, AxisNormal::My, AxisNormal::Mz];

    #[allow(dead_code)]
    pub fn from_u32(num: u32) -> Self{
        match num{
            0 => AxisNormal::X,
            1 => AxisNormal::Y,
            2 => AxisNormal::Z,
            3 => AxisNormal::Mx,
            4 => AxisNormal::My,
            5 => AxisNormal::Mz,
            _ => panic!("num out of bound!"),
        }
    }
    /// z
    #[allow(dead_code)]
    pub fn normal(&self) -> [f32; 3] {
        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as usize ;
        let sig = if n < 3 {1.0} else {-1.0};
        arr[n % 3] = sig;
        arr
    }
    
    /// x
    #[allow(dead_code)]
    pub fn tangent(&self) -> [f32; 3] {

        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as usize ;
        let sig = if n < 3 {1.0} else {-1.0};
        arr[(n + 1) % 3] = sig;
        arr
    }

    /// y
    #[allow(dead_code)]
    pub fn btangent(&self) -> [f32; 3] {

        let mut arr : [f32; 3] = [0.0; 3];
        let n = *self as usize ;
        // let sig = if n < 3 {1.0} else {-1.0};
        arr[(n + 2) % 3] = 1.0;
        arr
    }

    pub fn mat3(&self) -> [[f32; 3]; 3] {
        let mut arr : [[f32; 3]; 3] = [[0.0; 3]; 3];
        let n = *self as usize ;
        let sig = if n < 3 {1.0} else {-1.0};
        arr[0][(n + 1) % 3] = sig;
        arr[1][(n + 2) % 3] = 1.0;
        arr[2][n % 3]= sig;
        arr
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

#[cfg(test)]
mod test{
    use super::*;

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
            let mat3_2 =  Mat3::from_col_arrays(axis.mat3());
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