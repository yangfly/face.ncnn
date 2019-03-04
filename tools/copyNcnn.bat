
cd ..
set path=%cd%
copy %path%\3rdparty\src\ncnn\build\tools\caffe\Release\caffe2ncnn.exe %path%\3rdparty\bin\caffe2ncnn.exe
copy %path%\3rdparty\src\ncnn\build\install\lib\ncnn.lib %path%\3rdparty\lib\ncnn.lib

mkdir %path%\3rdparty\include\ncnn
copy %path%\3rdparty\src\ncnn\build\install\include\* %path%\3rdparty\include\ncnn

pause