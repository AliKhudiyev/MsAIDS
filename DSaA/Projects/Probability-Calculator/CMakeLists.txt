cmake_minimum_required(VERSION 3.10)

project(ProbabilityCalculator)

set(EXECUTABLE_NAME pc)
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(${EXECUTABLE_NAME} main.c ${SRCS})
