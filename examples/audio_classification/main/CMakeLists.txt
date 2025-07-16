# Main component for audio classification example

# Embed the model file
set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../../../fixed2conv_model.tflite")

idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES acoustics-porting freertos esp_timer esp_common
    PRIV_REQUIRES acoustics
    EMBED_FILES ${MODEL_FILE}
)
