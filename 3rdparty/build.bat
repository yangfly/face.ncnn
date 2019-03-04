:: build types can be (Release  Debug  Release,Debug)
set types=%1
if not defined types set types=Release,Debug
set path_root=%cd%
@echo Build protobuf and ncnn with types: %types%

for %%t in (%types%) do (call :sub_build %%t)
goto :eof

:sub_build
  set type=%1
  @echo  Building type: %type%
	set path_lib=%path_root%\lib\%type%
	if not exist %path_lib% mkdir %path_lib%
  set path_bin=%path_root%\bin\%type%
  if not exist %path_bin% mkdir %path_bin%
	
	:: build protobuf
	cd %path_root%\src\protobuf\cmake
	if exist build rmdir /S /Q build
	mkdir build
	cd build
	cmake .. -Dprotobuf_BUILD_TESTS=OFF ^
			 -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
			 -DCMAKE_BUILD_TYPE=%type% ^
			 -G "NMake Makefiles"
	nmake protoc
	: copy protobuf to 3rdparty
	call extract_includes.bat
  if exist %path_root%\include\google rmdir /S /Q %path_root%\include\google
	move include\google %path_root%\include\
	copy *.lib %path_lib%\
	copy *.exe %path_bin%\

	:: build ncnn
	cd %path_root%\src\ncnn
	if exist build rmdir /S /Q build
	mkdir build
	cd build
  if "%type%" == "Debug" (set ext=d) else (set ext=)
	cmake .. -DProtobuf_INCLUDE_DIR=%path_root%\include ^
			 -DProtobuf_LIBRARIES=%path_lib%\libprotobuf%ext%.lib ^
			 -DProtobuf_PROTOC_EXECUTABLE=%path_bin%\protoc.exe ^
			 -DCMAKE_INSTALL_PREFIX=%cd%\install ^
			 -DCMAKE_BUILD_TYPE=%type% ^
			 -G "NMake Makefiles"
	nmake
	nmake install
	: copy ncnn to 3rdparty
  if exist %path_root%\include\ncnn rmdir /S /Q %path_root%\include\ncnn
	mkdir %path_root%\include\ncnn
	copy %cd%\install\include\* %path_root%\include\ncnn\
	copy %cd%\install\lib\ncnn.lib %path_lib%\
  if "%type%" == "Debug" (copy %cd%\src\CMakeFiles\ncnn.dir\ncnn.pdb %path_lib%\)
	copy %cd%\tools\caffe\caffe2ncnn.exe %path_bin%\
	cd %path_root%
