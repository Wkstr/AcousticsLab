# Microphone Driver Component
# Following the same pattern as LIS3DHTR driver

set(MICROPHONE_DRIVER_SRCS
    "src/microphone.cpp"
)

set(MICROPHONE_DRIVER_INCLUDES
    "inc"
)

# Register as IDF component
idf_component_register(
    SRCS ${MICROPHONE_DRIVER_SRCS}
    INCLUDE_DIRS ${MICROPHONE_DRIVER_INCLUDES}
    REQUIRES driver esp_common freertos
    PRIV_REQUIRES esp_timer
)

# Add compile options for optimization
add_compile_options(-fdiagnostics-color=always -ffast-math -O2)

# Component-specific definitions
target_compile_definitions(${COMPONENT_LIB} PUBLIC
    MICROPHONE_DRIVER_VERSION_MAJOR=1
    MICROPHONE_DRIVER_VERSION_MINOR=0
    MICROPHONE_DRIVER_VERSION_PATCH=0
)
