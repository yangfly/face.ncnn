cd ..\3rdparty\src\ncnn
mkdir build
cd build
cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_CONFIGURATION_TYPES=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=%cd%/../../../include -DProtobuf_LIBRARIES=%cd%/../../../lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=%cd%/../../../bin/protoc.exe
pause