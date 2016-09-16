cmake_minimum_required(VERSION 3.6.2)

project(skbeam)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PythonExtensions REQUIRED)

add_subdirectory(skbeam/core/accumulators)
add_subdirectory(src)
