cmake_minimum_required(VERSION 3.12.4)

set(LITTLEFS_INC_DIR
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/bd
)

file(GLOB_RECURSE LITTLEFS_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/bd/*.c
)

idf_component_register(
    SRCS ${LITTLEFS_SRCS}
    INCLUDE_DIRS ${LITTLEFS_INC_DIR}
    REQUIRES driver freertos esp_partition
)
