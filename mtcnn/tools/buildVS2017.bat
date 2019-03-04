cd ..
mkdir VS2017
cd VS2017
cmake .. -G"Visual Studio 15 2017 Win64" -DCMAKE_CONFIGURATION_TYPES=Release
mkdir Release
copy ..\..\3rdparty\bin\*.dll Release
pause