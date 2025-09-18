cmake_minimum_required(VERSION 3.12.4)

set(PROJ_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
)

file(GLOB_RECURSE PROJ_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)

set(ACOUSTICS_API_VERSION "1" GLOBAL STRING "Acoustics API version")
set(ACOUSTICS_LIB_OPUS_ENABLE ON GLOBAL BOOL "Enable OPUS support in acoustics-porting")
set(ACOUSTICS_PORTING_LIB_TFLM_ENABLE ON GLOBAL BOOL "Enable TFLM support in acoustics-porting")
set(ACOUSTICS_PORTING_LIB_DL_FFT_ENABLE ON GLOBAL BOOL "Enable DL_FFT support in acoustics-porting")

set(SOUND_BOARD "" CACHE STRING "Board selection: empty(default)|respeaker_lite")
if(SOUND_BOARD STREQUAL "respeaker_lite")
    set(BOARD_RESPEAKER_LITE ON CACHE BOOL "Enable ReSpeaker Lite board support" FORCE)
    message(STATUS "Board for sound_sampler: ReSpeaker Lite enabled")
else()
    set(BOARD_RESPEAKER_LITE OFF CACHE BOOL "Enable ReSpeaker Lite board support" FORCE)
    message(STATUS "Board for sound_sampler: default (XIAO ESP32S3 mic)")
endif()

idf_component_register(
    SRCS ${PROJ_SRCS}
    INCLUDE_DIRS ${PROJ_INCLUDE_DIRS}
    REQUIRES freertos esp_psram esp_system acoustics-porting
)

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)
