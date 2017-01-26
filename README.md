# gpu_computing_in_robotics

This tutorial concenrs 3D lidar data processing with CUDA
Content of the tutorial:

lesson_0: basic transformations

lesson_1: down-sampling

lesson_2: noise removal (naive)

lesson_3: nearest neighborhood search

lesson_4: noise removal

lesson_5: normal vector computation

lesson_6: projections

lesson_7: basic semantics

lesson_8: semantic nearest neighborhood search

lesson_9: data registration Iterative Closest Point

lesson_10: data registration semantic Iterative Closest Point

lesson_11: data registration point to projection Iterative Closest Point

# requirements
Software was developed and tested on LINUX UBUNTU 14.04 with following libraries
OpenGL, GLUT, PCL 1.5, CUDA>=7.5

remark: there is a problem with PCL on UBUNTU 16.04 (work in progress)

# build
Each lesson is an independent software package, thus the following steps should be performed:
```
cd lesson_X
mkdir BUILD
cd BUILD
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./lesson_X
```
