# Main component for audio classification example

# Embed the model file
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../models/fixed2conv_model.tflite")

set(PROJ_INCLUDE_DIRS
    ${CMAKE_CURRENT_LIST_DIR}
)

file(GLOB_RECURSE PROJ_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)

idf_component_register(
    SRCS ${PROJ_SRCS}
    INCLUDE_DIRS ${PROJ_INCLUDE_DIRS}
    REQUIRES acoustics-porting freertos esp_timer esp_common
    PRIV_REQUIRES acoustics
    EMBED_FILES ${MODEL_FILE}
)
