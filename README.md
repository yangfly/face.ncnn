## 前言

在 [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) 基础上扩展支持 faceboxes 和 s3fd，windows 端采用 vs2017 构建。
> [ncnn](https://github.com/Tencent/ncnn) 是腾讯优图在七月份开源的，一款手机端极致优化的前向计算框架；开源有几个月了，仍然是开源里的扛把子（给nihui大佬递茶）。之前也测试移植过，这次主要做个整理，鉴于很多人只想在window下搭建并调试，本次主要基于MTCNN的人脸检测例子，进行一次该框架的搭建，构建流程主要采用脚本编写，目的在于简单演示下流程操作。

## 主要环境

- git
- cmake
- vs2017
- android studio

## 前期准备

使用 windows 的 pc 端调试，繁琐的是依赖库的生成和引用。依赖库中 Protobuf 是 caffe 的模型序列化存储的规则库，将 caffe 框架转 ncnn 框架模型用到，另外 opencv 库主要用于范例的图像读取操作，可自己配置，或直接使用个人在 3rdparty 文件夹下编译好的库；

### 1. 下载源码

下载源码并更新子模块, protobuf 源码库比较大，更新会比较慢

```
git clone https://github.com/yangfly/face.ncnn.git --recursive
```

如果你在 clone 时忘记加 `--recursive`, 可以使用 `git submodule update --init` 更新子模块

### 2. 编译 protobuf

- 双击 tools 下的 protobuf.bat 脚本在 3rdparty/src/protobuf/cmake/build 下生成 protobuf.sln 工程;
- 使用 VS2017 打开 protobuf.sln 工程，右键 `ALL_BUILD` 生成 Debug X64 及 Release X64 版本；
- 双击 tools 下的 copyProtobuf.bat 脚本，拷贝 protobuf 的依赖库到第三方公共文件夹 3rdparty 下。

### 3. 编译 ncnn

- 双击 tools 下的 ncnn.bat 脚本在 3rdparty/src/ncnn/build 下生成 ncnn.sln 工程;
- 使用 VS2017 打开 ncnn.sln 工程，右键 `install` 安装 Release X64 版本；
- 双击 tools 下的 copyNcnn.bat 脚本，拷贝 ncnn 的依赖库到第三方公共文件夹 3rdparty 下。

build 目录下生成的 src 中包含 ncnn.lib 库，tools 里有 caffe 以及 mxnet 的转换工具；
copyNcnn.bat 脚本主要拷贝了 caffe2ncnn 的 exe 文件，ncnn 的 lib 库及 .h 头文件；

## 测试算法

- [mtcnn](mtcnn/README.md)

## 参考和感谢

- [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) by @moli232777144
- [ncnn](https://github.com/Tencent/ncnn) by @Tencent
