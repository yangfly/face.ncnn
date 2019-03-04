## 前言

在 [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) 基础上扩展支持 faceboxes 和 s3fd，windows 端采用 vs2017 构建。
> [ncnn](https://github.com/Tencent/ncnn) 是腾讯优图在七月份开源的，一款手机端极致优化的前向计算框架；开源有几个月了，仍然是开源里的扛把子（给nihui大佬递茶）。之前也测试移植过，这次主要做个整理，鉴于很多人只想在window下搭建并调试，本次主要基于MTCNN的人脸检测例子，进行一次该框架的搭建，构建流程主要采用脚本编写，目的在于简单演示下流程操作。

## 主要环境

- git
- cmake
- vs2017
- android studio

### 1. 下载源码

下载源码并更新子模块, protobuf 源码库比较大，更新会比较慢

```
git clone https://github.com/yangfly/face.ncnn.git --recursive
```

如果你在 clone 时忘记加 `--recursive`, 可以使用 `git submodule update --init` 更新子模块

### 2. 编译三方库

- protobuf: 用于 caffe2ncnn.exe 转模型过程中用于解析 caffe 模型;
- ncnn: 用于高效的模型推理库;
- opencv: 用于 demo 中图片读写和结果可视化。

因为代码中已经包含了编译好的 opencv 库，所以接下来只需要编译 protobuf 和 ncnn，为了避免繁琐笨重的 Visual Studio 编译，我们选择 cmake + nmake 来从命令行自动连续构建 protobuf 和 ncnn。

操作步骤如下：

1. 打开 VS 命令行窗口，注意不可以用普通命令行窗口，VS2015 也可以找到相应入口
   `Start` → `Programs` → `Visual Studio 2017` → `Visual Studio Tools` → `x64 Native Tools Command Prompt for VS 2017`；
2. 在命令行进入 `3rdparty` 目录，执行编译脚本，`build.bat` 在不带参数情况下，默认为 `Release,Debug` 模式，可以只构建 `Release` 版本以节省编译时间。
```
cd <path_to_face.ncnn>/3rdparty
build.bat [Release,Debug  Release  Debug]
```

## 编译 mtcnn

- [编译测试 mtcnn](mtcnn/README.md)

## 参考和感谢

- [mtcnn_ncnn](https://github.com/moli232777144/mtcnn_ncnn) by @moli232777144
- [ncnn](https://github.com/Tencent/ncnn) by @Tencent
