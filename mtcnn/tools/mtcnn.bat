: set generator="Visual Studio 14 2015 Win64"
set generator="Visual Studio 15 2017 Win64"

cd ..
set path_root=%cd%\..
set path_3rdparty=%path_root%\3rdparty

if exist %path_3rdparty%\lib\Release (call :release_exist) ^
else (if exist %path_3rdparty%\lib\Debug (set build_type=Debug))
echo %build_type%

if exist build rmdir /S /Q build
mkdir build
cd build
cmake .. -G%generator% -DCMAKE_CONFIGURATION_TYPES=%build_type%
pause
goto :eof

:release_exist
  set build_type=Release
  if exist %path_3rdparty%\lib\Debug (set build_type=%build_type%;Debug)

:: for %%t in (%build_type%) do (call :copy_bin %%t)
:: goto :eof

:: :copy_bin
::   set type=%1
::   mkdir %type%
::   copy %path_root%\3rdparty\bin\%type%\*.dll %type%\
::   copy %path_root%\3rdparty\bin\*.dll %type%\
:: 