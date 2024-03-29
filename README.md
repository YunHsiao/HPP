Heterogeneous Parallel Programming Course Assignments
=====

Based on the original [WebGPU](https://github.com/abduld/WebGPU) website repo,<br>
With these necessary adjustments:
* Rearraged file structure
* Fixed cmake script
* Fixed runmp bash script
* Fixed all markdown links

Basically everything now works as they originally designed to.<br>
Since the original website has been shutdown for good, everything here is self-contained.<br>
There are enough informations to go through all the assignments comfortably.

To get started, build the cmake project as the following chapters describe.<br>
Tested only on Windows (Windows 10 1903, VS 2017, CUDA 10.1)

Note that the MPs is offered in an inconsistant(and even fragmented) order with the course material,<br>
so it is best to progress in the following order:

Week | Assignment | Content
:--: | :--: | :--
Week 0 | MP0 | Device Query
Week 1 | MP1 | Vector Addition
Week 1 | MP2 | Basic Matrix-Matrix Multiplication
Week 2 | MP3 | Tiled Matrix-Matrix Multiplication
Week 3 | MP6 | Image Convolution
Week 4 | MP4 | List Reduction
Week 4 | MP5 | List Scan
Week 5 | MP11 | Histogram Equalization
Week 6 | MP12 | Streamed Vector Addition
Week 7 | MP7 | (Optional) OpenCL Vector Addition
Week 7 | MP9 | (Optional) OpenACC Vector Addition
Week 8 | MP8 | (Optional) C++ AMP Vector Addition

Reference implementation of all the above assignments can be found at [the `ref` branch](https://github.com/YunHsiao/HPP/tree/ref).

libwb
=====

## Compiling and Running on Linux and Windows
### by Tran Minh Quan

This is a tutorial explains how to compile and run your Machine
Problems (MPs) offline **without separating on building libwb.**

_Caution: **If you don't have NVIDIA GPUs ([CUDA Capable GPU](https://developer.nvidia.com/cuda-gpus)s) on your local machine, you cannot run the executable binaries.**_

First, regardless your platform is, please install CUDA 5.5
and Cmake 2.8 ([](http://www.cmake.org/)[](http://www.cmake.org/)[](http://www.cmake.org/)[](http://www.cmake.org/)[http://www.cmake.org/](http://www.cmake.org/)) , then set the path appropriate to these things (linux).

Check out the source codes (only skeleton codes) for MPs as
following

[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[](https://github.com/hvcl/hetero13)[https://github.com/hvcl/hetero13](https://github.com/hvcl/hetero13)

    git clone https://github.com/abduld/libwb

1\. If you are under Linux environment, you should use gcc lower than 4.7 (mine is 4.4.7).
Ortherwise, it will not be compatible with nvcc

    cd libwb
    ls
    mkdir build
    cd build/
    cmake ..
    make -j4
    ./MP0

2\. If you are under Windows environment

Open Cmake Gui:

    Where is the source code: {libwb}/
    Where to build the binary: {libwb}/build

![image](https://coursera-forum-screenshots.s3.amazonaws.com/5d/d77a10785611e3ae687ff4063e578b/1.png)

Press Configure, Yes and choose your compiler (in this case Visual
Studio 10 (32 bit) or Visual Studio 10 Win64 (64 bit), then press Finish

![image](https://coursera-forum-screenshots.s3.amazonaws.com/75/ee29f0785611e3ae687ff4063e578b/2.png)

![image](https://coursera-forum-screenshots.s3.amazonaws.com/e5/1e0fc0785611e3ae687ff4063e578b/3.png)

Press Configure one more time and generate

![image](https://coursera-forum-screenshots.s3.amazonaws.com/11/315360785711e3ae687ff4063e578b/4.png)

Open your generated folder and Double click on libwb.sln

![image](https://coursera-forum-screenshots.s3.amazonaws.com/3a/5da3b0785711e3ae687ff4063e578b/5.png)

Right click to MP0 and click "set as startup project"

Press Ctrl F5

Whenever you do the MPs, change the MP accordingly.

Best regards,

P/s1: Sorry about the name of project, it should be hetero14 but I forget that we are already in the new year. My appologize :-P

P/s2: If you are using MAC, please consider reading this link and modify your CMakeLists.txt
[https://class.coursera.org/hetero-002/forum/thread?thread\_id=83](https://class.coursera.org/hetero-002/forum/thread?thread_id=83)
