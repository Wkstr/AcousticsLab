# Audio Classification Example

cmake_minimum_required(VERSION 3.16)

# Set component directories before including project.cmake
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../components"
)

# Set the project name
set(PROJECT_NAME "audio_classification")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Include ESP-IDF build system
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Define the project
project(${PROJECT_NAME})

# Add compile options
add_compile_options(-fdiagnostics-color=always -ffast-math -O2)