## introduction
 基于wgpu的球面地形渲染，基础思想是cube-sphere和四叉树
 
## project structure
   

## save directory structure
/saves
 world.ron
 planet_A/
  planet_A.ron
  planet_A.terrain
  ..
 planet_B/
  planet_B.ron
  planet_B.terrain

## 算法
### cube-sphere
    cube-sphere 是一种表达球面的方法，即将一个立方体的六个面映射到球面上。对于立方体表面任意一点P，可通过公式映射到球面：P.normalized()，其中P.normalied()表示P的单位向量。然而，该映射方式会使得棱角附近的点映射后会挤压在一起，因此需要对映射前的点进行一定的变换，使得映射后的点更加均匀分布。这里采用的是COBE变换。

