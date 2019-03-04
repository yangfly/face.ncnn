## 前言

[MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) 是一个快速的人脸检测和对齐算法。

## 前期准备

在进行算法测试之前，请先按照 **[说明](../README.md)** 编译 protobuf 和 ncnn 三方库。


## 转换 MTCNN 模型 (可选)

**mtcnn/models 下已默认提供了转换好的模型。**

MTCNN 官方模型使用 Caffe 的 Mablab 接口训练的，是列优先 (col-major) 模型，与 ncnn 默认的行优先 (row-major) 模型格式不匹配，因此需要进行格式转换。我们在 models 目录下提供了转换后的行优先模型。ncnn 的模型转换方法是：

```
caffe2ncnn.exe xx.prototxt xx.caffemodel xx.param xx.bin
```

因此可以方便地双击 tools 下的 caffe2ncnn.bat 脚本，一次性转换 [Mtcnnv2](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv2/model) 的四个模型。

> 参考 [ElegantGod的ncnn](https://github.com/ElegantGod/ncnn) 的 ncnn 改进，提取了其中转化准则文件，放 tools 目录下的 caffe2ncnn.cpp 文件，接着替换 ncnn 的 tools/caffe 同文件，重新生成 caffe2ncnn.exe，并依次执行一次以上模型转换步骤。

## 编译 MTCNN

- 双击 tools 下的 mtcnn.bat 脚本在 build 下生成 mtcnn.sln 工程;
- 使用 VS2017 打开 mtcnn.sln 工程，右键 `mtcnn` 设为启动项目；
- 快捷键 `Ctrl + F5` 快速生成 Release X64 版本并运行。

附粗略实测结果：

CPU 单核：Intel(R) Core(TM) i5-4590 @ 3.30GHz

  模式   |   时间
:------: | :------:
 w LNet  | 52.26 ms
w/o LNet | 47.66 ms
<!-- 

#  安卓端调试 (暂未调试)：

ncnn的安卓端源码范例主要采用的mk文件构造，win开发安卓端大家通常使用AS的cmake来构造工程，下面主要简单介绍相关流程，具体细节参考mtcnnn-AS工程；

1. 新建工程
参考网上配置andorid studio的c++混编环境，新建一个mtcnn—AS的工程；

2. 配置相关文件位置（ps：最新的lib会更快）
	- 下载ncnn的release里的[安卓端lib](https://github.com/Tencent/ncnn/releases)，或者调用tools/build_android.bat
	- 将arm端的.a文件放至相关jniLibs对应目录下；
	- include的头文件放至cpp目录下；
	- 将mtcnn的c++的接口文件放在cpp目录下；

3. 新建jni接口文件，相关方法自行参考网上其他教程；

4. CmakeList文件的编写：

```
cmake_minimum_required(VERSION 3.4.1)

#include头文件目录
include_directories(src/main/cpp/include
                    src/main/cpp/)

#source directory源文件目录
file(GLOB MTCNN_SRC src/main/cpp/*.h
                    src/main/cpp/*.cpp)
set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

#添加ncnn库
add_library(libncnn STATIC IMPORTED )
set_target_properties(libncnn
  PROPERTIES IMPORTED_LOCATION
  ${CMAKE_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/libncnn.a)

#编译为动态库
add_library(mtcnn SHARED ${MTCNN_COMPILE_CODE})

#添加工程所依赖的库
find_library(  log-lib log )
target_link_libraries(  mtcnn
                       libncnn
                       jnigraphics
                       z
                       ${log-lib} )

```
5.成功编译mtcnn的so库，在安卓的MainActivity编写接口使用的相关操作；

ps:android6.0以上机型，部分会出现模型读写到sd卡因权限失败问题;
若所有图像均未检测到人脸，请检测下sd下是否存储模型的mtcnn目录。

附粗略实测时间（高通625）：


安卓端速度  | 时间
---|---
squeezenet（原始例子）| 121ms
mtcnn（最小人脸40）| 47ms -->

## 参考和感谢

- [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) by @moli232777144
- [ncnn](https://github.com/Tencent/ncnn) by @Tencent
