cmake_minimum_required(VERSION 3.12.4)

set(LIS3DHTR_INC_DIR ${CMAKE_CURRENT_LIST_DIR}/inc)

file(GLOB_RECURSE LIS3DHTR_SRCS
    src/*.c
    src/*.cpp
)

idf_component_register(
    SRCS ${LIS3DHTR_SRCS}
    INCLUDE_DIRS ${LIS3DHTR_INC_DIR}
    REQUIRES driver freertos
)
