# Audio Classification Example

cmake_minimum_required(VERSION 3.16)

# Set target to ESP32-S3 for I2S PDM support
set(IDF_TARGET esp32s3)

# Set component directories before including project.cmake
set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_SOURCE_DIR}/../../components"
)

# Set ACOUSTIC_SDK_TARGET before project() call
set_property(GLOBAL PROPERTY ACOUSTIC_SDK_TARGET "POSIX")

# Set the project name
set(PROJECT_NAME "audio_classification")

# Include ESP-IDF build system
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# Define the project
project(${PROJECT_NAME})

# Add compile options
add_compile_options(-fdiagnostics-color=always -ffast-math -O2)

# Model is loaded from absolute path, no copying needed
