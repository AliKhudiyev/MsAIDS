# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.18.0/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.18.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build

# Include any dependencies generated for this target.
include CMakeFiles/dea.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dea.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dea.dir/flags.make

CMakeFiles/dea.dir/src/dea.cpp.o: CMakeFiles/dea.dir/flags.make
CMakeFiles/dea.dir/src/dea.cpp.o: ../src/dea.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dea.dir/src/dea.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dea.dir/src/dea.cpp.o -c /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/src/dea.cpp

CMakeFiles/dea.dir/src/dea.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dea.dir/src/dea.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/src/dea.cpp > CMakeFiles/dea.dir/src/dea.cpp.i

CMakeFiles/dea.dir/src/dea.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dea.dir/src/dea.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/src/dea.cpp -o CMakeFiles/dea.dir/src/dea.cpp.s

# Object files for target dea
dea_OBJECTS = \
"CMakeFiles/dea.dir/src/dea.cpp.o"

# External object files for target dea
dea_EXTERNAL_OBJECTS =

dea: CMakeFiles/dea.dir/src/dea.cpp.o
dea: CMakeFiles/dea.dir/build.make
dea: /usr/local/lib/libomp.dylib
dea: CMakeFiles/dea.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dea"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dea.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dea.dir/build: dea

.PHONY : CMakeFiles/dea.dir/build

CMakeFiles/dea.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dea.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dea.dir/clean

CMakeFiles/dea.dir/depend:
	cd /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build /Users/alikhudiyev/Desktop/MsAIDS/Project-Management/Differential-Evolution+Genetic/build/CMakeFiles/dea.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dea.dir/depend

