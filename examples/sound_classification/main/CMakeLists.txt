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

if(NOT DEFINED PORTING_BOARD_MODEL_RESPEAKER_LITE)
    set(PORTING_BOARD_MODEL_RESPEAKER_LITE 0)
endif()
if(NOT DEFINED PORTING_BOARD_MODEL_XIAO_S3)
    set(PORTING_BOARD_MODEL_XIAO_S3 1) 
endif()

if(PORTING_BOARD_MODEL_RESPEAKER_LITE)
    message(STATUS "Board: ReSpeaker Lite enabled")
elseif(PORTING_BOARD_MODEL_XIAO_S3)
    message(STATUS "Board: XIAO ESP32S3 enabled (default)")
else()
    message(STATUS "Board: No specific board model enabled")
endif()

idf_component_register(
    SRCS ${PROJ_SRCS}
    INCLUDE_DIRS ${PROJ_INCLUDE_DIRS}
    REQUIRES freertos esp_psram esp_system acoustics-porting
)

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)
