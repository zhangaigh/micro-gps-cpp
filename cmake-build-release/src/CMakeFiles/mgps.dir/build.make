# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release

# Include any dependencies generated for this target.
include src/CMakeFiles/mgps.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/mgps.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/mgps.dir/flags.make

src/CMakeFiles/mgps.dir/image_dataset.cpp.o: src/CMakeFiles/mgps.dir/flags.make
src/CMakeFiles/mgps.dir/image_dataset.cpp.o: ../src/image_dataset.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/mgps.dir/image_dataset.cpp.o"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mgps.dir/image_dataset.cpp.o -c /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_dataset.cpp

src/CMakeFiles/mgps.dir/image_dataset.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mgps.dir/image_dataset.cpp.i"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_dataset.cpp > CMakeFiles/mgps.dir/image_dataset.cpp.i

src/CMakeFiles/mgps.dir/image_dataset.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mgps.dir/image_dataset.cpp.s"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_dataset.cpp -o CMakeFiles/mgps.dir/image_dataset.cpp.s

src/CMakeFiles/mgps.dir/image_dataset.cpp.o.requires:

.PHONY : src/CMakeFiles/mgps.dir/image_dataset.cpp.o.requires

src/CMakeFiles/mgps.dir/image_dataset.cpp.o.provides: src/CMakeFiles/mgps.dir/image_dataset.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/mgps.dir/build.make src/CMakeFiles/mgps.dir/image_dataset.cpp.o.provides.build
.PHONY : src/CMakeFiles/mgps.dir/image_dataset.cpp.o.provides

src/CMakeFiles/mgps.dir/image_dataset.cpp.o.provides.build: src/CMakeFiles/mgps.dir/image_dataset.cpp.o


src/CMakeFiles/mgps.dir/image.cpp.o: src/CMakeFiles/mgps.dir/flags.make
src/CMakeFiles/mgps.dir/image.cpp.o: ../src/image.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/mgps.dir/image.cpp.o"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mgps.dir/image.cpp.o -c /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image.cpp

src/CMakeFiles/mgps.dir/image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mgps.dir/image.cpp.i"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image.cpp > CMakeFiles/mgps.dir/image.cpp.i

src/CMakeFiles/mgps.dir/image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mgps.dir/image.cpp.s"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image.cpp -o CMakeFiles/mgps.dir/image.cpp.s

src/CMakeFiles/mgps.dir/image.cpp.o.requires:

.PHONY : src/CMakeFiles/mgps.dir/image.cpp.o.requires

src/CMakeFiles/mgps.dir/image.cpp.o.provides: src/CMakeFiles/mgps.dir/image.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/mgps.dir/build.make src/CMakeFiles/mgps.dir/image.cpp.o.provides.build
.PHONY : src/CMakeFiles/mgps.dir/image.cpp.o.provides

src/CMakeFiles/mgps.dir/image.cpp.o.provides.build: src/CMakeFiles/mgps.dir/image.cpp.o


src/CMakeFiles/mgps.dir/image_func.cpp.o: src/CMakeFiles/mgps.dir/flags.make
src/CMakeFiles/mgps.dir/image_func.cpp.o: ../src/image_func.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/mgps.dir/image_func.cpp.o"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mgps.dir/image_func.cpp.o -c /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_func.cpp

src/CMakeFiles/mgps.dir/image_func.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mgps.dir/image_func.cpp.i"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_func.cpp > CMakeFiles/mgps.dir/image_func.cpp.i

src/CMakeFiles/mgps.dir/image_func.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mgps.dir/image_func.cpp.s"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/image_func.cpp -o CMakeFiles/mgps.dir/image_func.cpp.s

src/CMakeFiles/mgps.dir/image_func.cpp.o.requires:

.PHONY : src/CMakeFiles/mgps.dir/image_func.cpp.o.requires

src/CMakeFiles/mgps.dir/image_func.cpp.o.provides: src/CMakeFiles/mgps.dir/image_func.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/mgps.dir/build.make src/CMakeFiles/mgps.dir/image_func.cpp.o.provides.build
.PHONY : src/CMakeFiles/mgps.dir/image_func.cpp.o.provides

src/CMakeFiles/mgps.dir/image_func.cpp.o.provides.build: src/CMakeFiles/mgps.dir/image_func.cpp.o


src/CMakeFiles/mgps.dir/micro_gps.cpp.o: src/CMakeFiles/mgps.dir/flags.make
src/CMakeFiles/mgps.dir/micro_gps.cpp.o: ../src/micro_gps.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/mgps.dir/micro_gps.cpp.o"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mgps.dir/micro_gps.cpp.o -c /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/micro_gps.cpp

src/CMakeFiles/mgps.dir/micro_gps.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mgps.dir/micro_gps.cpp.i"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/micro_gps.cpp > CMakeFiles/mgps.dir/micro_gps.cpp.i

src/CMakeFiles/mgps.dir/micro_gps.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mgps.dir/micro_gps.cpp.s"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/micro_gps.cpp -o CMakeFiles/mgps.dir/micro_gps.cpp.s

src/CMakeFiles/mgps.dir/micro_gps.cpp.o.requires:

.PHONY : src/CMakeFiles/mgps.dir/micro_gps.cpp.o.requires

src/CMakeFiles/mgps.dir/micro_gps.cpp.o.provides: src/CMakeFiles/mgps.dir/micro_gps.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/mgps.dir/build.make src/CMakeFiles/mgps.dir/micro_gps.cpp.o.provides.build
.PHONY : src/CMakeFiles/mgps.dir/micro_gps.cpp.o.provides

src/CMakeFiles/mgps.dir/micro_gps.cpp.o.provides.build: src/CMakeFiles/mgps.dir/micro_gps.cpp.o


src/CMakeFiles/mgps.dir/inpolygon.c.o: src/CMakeFiles/mgps.dir/flags.make
src/CMakeFiles/mgps.dir/inpolygon.c.o: ../src/inpolygon.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object src/CMakeFiles/mgps.dir/inpolygon.c.o"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/mgps.dir/inpolygon.c.o   -c /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/inpolygon.c

src/CMakeFiles/mgps.dir/inpolygon.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/mgps.dir/inpolygon.c.i"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/inpolygon.c > CMakeFiles/mgps.dir/inpolygon.c.i

src/CMakeFiles/mgps.dir/inpolygon.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/mgps.dir/inpolygon.c.s"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src/inpolygon.c -o CMakeFiles/mgps.dir/inpolygon.c.s

src/CMakeFiles/mgps.dir/inpolygon.c.o.requires:

.PHONY : src/CMakeFiles/mgps.dir/inpolygon.c.o.requires

src/CMakeFiles/mgps.dir/inpolygon.c.o.provides: src/CMakeFiles/mgps.dir/inpolygon.c.o.requires
	$(MAKE) -f src/CMakeFiles/mgps.dir/build.make src/CMakeFiles/mgps.dir/inpolygon.c.o.provides.build
.PHONY : src/CMakeFiles/mgps.dir/inpolygon.c.o.provides

src/CMakeFiles/mgps.dir/inpolygon.c.o.provides.build: src/CMakeFiles/mgps.dir/inpolygon.c.o


# Object files for target mgps
mgps_OBJECTS = \
"CMakeFiles/mgps.dir/image_dataset.cpp.o" \
"CMakeFiles/mgps.dir/image.cpp.o" \
"CMakeFiles/mgps.dir/image_func.cpp.o" \
"CMakeFiles/mgps.dir/micro_gps.cpp.o" \
"CMakeFiles/mgps.dir/inpolygon.c.o"

# External object files for target mgps
mgps_EXTERNAL_OBJECTS =

src/libmgps.a: src/CMakeFiles/mgps.dir/image_dataset.cpp.o
src/libmgps.a: src/CMakeFiles/mgps.dir/image.cpp.o
src/libmgps.a: src/CMakeFiles/mgps.dir/image_func.cpp.o
src/libmgps.a: src/CMakeFiles/mgps.dir/micro_gps.cpp.o
src/libmgps.a: src/CMakeFiles/mgps.dir/inpolygon.c.o
src/libmgps.a: src/CMakeFiles/mgps.dir/build.make
src/libmgps.a: src/CMakeFiles/mgps.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX static library libmgps.a"
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/mgps.dir/cmake_clean_target.cmake
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mgps.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/mgps.dir/build: src/libmgps.a

.PHONY : src/CMakeFiles/mgps.dir/build

src/CMakeFiles/mgps.dir/requires: src/CMakeFiles/mgps.dir/image_dataset.cpp.o.requires
src/CMakeFiles/mgps.dir/requires: src/CMakeFiles/mgps.dir/image.cpp.o.requires
src/CMakeFiles/mgps.dir/requires: src/CMakeFiles/mgps.dir/image_func.cpp.o.requires
src/CMakeFiles/mgps.dir/requires: src/CMakeFiles/mgps.dir/micro_gps.cpp.o.requires
src/CMakeFiles/mgps.dir/requires: src/CMakeFiles/mgps.dir/inpolygon.c.o.requires

.PHONY : src/CMakeFiles/mgps.dir/requires

src/CMakeFiles/mgps.dir/clean:
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src && $(CMAKE_COMMAND) -P CMakeFiles/mgps.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/mgps.dir/clean

src/CMakeFiles/mgps.dir/depend:
	cd /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2 /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/src /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src /Users/lgzhang/projects/micro_gps/code/mgps-cpp-v2/cmake-build-release/src/CMakeFiles/mgps.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/mgps.dir/depend
