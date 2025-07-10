cmake_minimum_required(VERSION 3.12.4)

message(STATUS "Configuring AcousticsLab SDK")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(ACOUSTIC_INCLUDES_DIR
    ${CMAKE_CURRENT_LIST_DIR}/
)
message(STATUS "Include directory: ${ACOUSTIC_INCLUDES_DIR}")

file(GLOB ACOUSTIC_SRCS_API
    ${CMAKE_CURRENT_LIST_DIR}/api/*.c
    ${CMAKE_CURRENT_LIST_DIR}/api/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/api/*.cpp
)

if(NOT DEFINED ACOUSTIC_API_VERSION)
    message(WARNING "ACOUSTIC_API_VERSION is not defined, defaulting to 0")
    set(ACOUSTIC_API_VERSION "0")
endif()

if(${ACOUSTIC_API_VERSION} STREQUAL "0")
    file(GLOB_RECURSE ACOUSTIC_SRCS_API_V0
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.c
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.hpp
        ${CMAKE_CURRENT_LIST_DIR}/api/v0/*.cpp
    )
    list(APPEND ACOUSTIC_SRCS_API ${ACOUSTIC_SRCS_API_V0})
    message(STATUS "Using API version 0 sources")
endif()

if(NOT DEFINED ACOUSTIC_SRCS_API)
    message(FATAL_ERROR "ACOUSTIC_SRCS_API is not defined")
endif()
message(STATUS "ACOUSTIC_SRCS_API: ${ACOUSTIC_SRCS_API}")

file(GLOB_RECURSE ACOUSTIC_SRCS_CORE
    ${CMAKE_CURRENT_LIST_DIR}/core/*.c
    ${CMAKE_CURRENT_LIST_DIR}/core/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/core/*.cpp
)
message(STATUS "ACOUSTIC_SRCS_CORE: ${ACOUSTIC_SRCS_CORE}")

file(GLOB_RECURSE ACOUSTIC_SRCS_HAL
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.c
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hal/*.cpp
)
message(STATUS "ACOUSTIC_SRCS_HAL: ${ACOUSTIC_SRCS_HAL}")

set(ACOUSTIC_SRCS
    ${ACOUSTIC_SRCS_API}
    ${ACOUSTIC_SRCS_CORE}
    ${ACOUSTIC_SRCS_HAL}
)

get_property(ACOUSTIC_SDK_TARGET GLOBAL PROPERTY ACOUSTIC_SDK_TARGET)
message(STATUS "ACOUSTIC_SDK_TARGET: ${ACOUSTIC_SDK_TARGET}")

if(NOT DEFINED ACOUSTIC_SDK_TARGET)
    message(FATAL_ERROR "ACOUSTIC_SDK_TARGET is not defined")
endif()

if(${ACOUSTIC_SDK_TARGET} STREQUAL "POSIX")
    message(STATUS "Building for POSIX")

    add_library(acoustics STATIC ${ACOUSTIC_SRCS})

    target_include_directories(acoustics PUBLIC
        ${ACOUSTIC_INCLUDES_DIR}
    )

    target_compile_definitions(acoustics PUBLIC
        ACOUSTICS_API_VERSION=${ACOUSTIC_API_VERSION}
    )

    set_target_properties(acoustics PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
endif()

add_compile_options(-fdiagnostics-color=always -ffast-math -O2)
