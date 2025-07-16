# AcousticsLab Audio Classification Pipeline Documentation

## Overview

This document describes the complete audio classification pipeline in AcousticsLab, including data flow, component interactions, and architectural highlights.

## Architecture Overview

```
Audio Input (I2S) → Sensor → Feature Extractor → TFLite Engine → Classification Results
     ↓                ↓            ↓                ↓                    ↓
  44032 samples   Float32     STFT Features    Quantized Int8      Probabilities
  (16-bit PCM)    Tensor      (43×232=9976)    Inference          (3 classes)
```

## Complete Data Flow

### 1. Audio Capture
**File**: `components/acoustics/hal/sensors/microphone_i2s.cpp`
- **Function**: `process()` method
- **Input**: I2S audio stream
- **Output**: 44032 samples of 16-bit PCM audio
- **Data Format**: Raw audio samples at 44032 Hz sampling rate

### 2. Audio Preprocessing
**File**: `examples/audio_classification/main/main.cpp`
- **Function**: `AudioClassificationDemo::processAudioChunk()`
- **Process**: Convert int16 PCM to float32 and normalize
- **Data Transformation**: 
  ```cpp
  float normalized_sample = static_cast<float>(pcm_sample) / 32768.0f;
  ```

### 3. Feature Extraction
**File**: `components/acoustics/hal/processors/feature_extractor.cpp`
- **Function**: `ProcessorFeatureExtractor::process()`
- **Algorithm**: Short-Time Fourier Transform (STFT) + Log Magnitude
- **Configuration**:
  - Frame Length: 2048 samples
  - Hop Length: 1024 samples  
  - FFT Size: 2048
  - Features per Frame: 232
  - Number of Frames: 43
- **Output**: 43 × 232 = 9976 log-magnitude spectral features

#### Feature Extraction Details:
1. **Windowing**: Apply Hanning window to each frame
2. **FFT**: KissFFT real-to-complex transform
3. **Magnitude**: Calculate |FFT|² power spectrum
4. **Log Transform**: log(magnitude + ε) for numerical stability
5. **Normalization**: Global mean-variance normalization

### 4. Model Inference
**File**: `components/acoustics/hal/engines/tflite_engine.cpp`
- **Function**: `TFLiteEngine::forward()`
- **Model**: `fixed2conv_model.tflite` (embedded binary)
- **Input Quantization**: Float32 → Int8
  ```cpp
  int8_t quantized = (int32_t)round(float_val / input_scale) + input_zero_point;
  ```
- **Inference**: TensorFlow Lite Micro execution
- **Output Dequantization**: Int8 → Float32
  ```cpp
  float score = (quantized_val - output_zero_point) * output_scale;
  ```

### 5. Classification Results
**File**: `examples/audio_classification/main/main.cpp`
- **Function**: `AudioClassificationDemo::run()`
- **Output**: Probability scores for 3 classes:
  - Class 0: "Background Noise"
  - Class 1: "one"
  - Class 2: "two"

## Key Components and Files

### Core Pipeline Components

1. **Inference Pipeline** (`components/acoustics/api/inference_pipeline.cpp`)
   - Orchestrates the entire pipeline
   - Manages tensor allocation and data flow
   - **Key Feature**: Dynamic tensor sizing based on component specifications

2. **Feature Extractor** (`components/acoustics/hal/processors/feature_extractor.cpp`)
   - STFT-based spectral feature extraction
   - **Key Feature**: Two-level memory allocation (PSRAM/Internal RAM)
   - **Key Feature**: Configurable frame count for model compatibility

3. **TFLite Engine** (`components/acoustics/hal/engines/tflite_engine.cpp`)
   - TensorFlow Lite Micro inference
   - **Key Feature**: Automatic quantization/dequantization handling
   - **Key Feature**: Memory-efficient model loading

4. **I2S Microphone Sensor** (`components/acoustics/hal/sensors/microphone_i2s.cpp`)
   - Hardware audio capture
   - **Key Feature**: Configurable I2S parameters
   - **Key Feature**: Circular buffer management

### Configuration and Metadata

1. **Feature Generator Config** (`main/feature_generator.h`)
   ```cpp
   #define CONFIG_NUM_FRAMES 43
   #define CONFIG_FEATURES_PER_FRAME 232
   #define CONFIG_OUTPUT_SIZE (CONFIG_NUM_FRAMES * CONFIG_FEATURES_PER_FRAME)
   ```

2. **Model Metadata** (`main/model_metadata.h`)
   - Input/output tensor specifications
   - Quantization parameters
   - Class labels

## Data Structures and Memory Management

### Tensor Management
**File**: `components/acoustics/core/tensor.hpp`
- **Key Feature**: RAII-based memory management
- **Key Feature**: Type-safe tensor operations
- **Key Feature**: Automatic reshape validation

### Memory Allocation Strategy
1. **PSRAM Usage**: Large buffers (FFT, features) allocated in PSRAM
2. **Internal RAM**: Small, frequently accessed data in fast memory
3. **Heap Caps**: ESP32-specific memory allocation with capability flags

## Architectural Highlights

### 1. Component-Based Architecture
- **Modular Design**: Each processing stage is an independent component
- **Interface Standardization**: Common `Processor` and `Engine` base classes
- **Configuration Management**: Unified config system across components

### 2. Memory Efficiency
- **Zero-Copy Operations**: Direct tensor data access without copying
- **Lazy Initialization**: Components initialize only when needed
- **Resource Cleanup**: Automatic resource management with RAII

### 3. Error Handling
- **Status-Based Returns**: Comprehensive error reporting with `core::Status`
- **Graceful Degradation**: Fallback mechanisms for memory allocation
- **Logging Integration**: Detailed logging at multiple levels

### 4. Performance Optimizations
- **KissFFT Integration**: Optimized FFT implementation for embedded systems
- **Quantized Inference**: Int8 quantization for faster inference
- **Buffer Reuse**: Minimize memory allocations during runtime

## Configuration Parameters

### Audio Processing
- **Sample Rate**: 44032 Hz
- **Bit Depth**: 16-bit PCM
- **Frame Size**: 2048 samples (~46.5ms at 44032 Hz)
- **Hop Size**: 1024 samples (~23.3ms overlap)

### Feature Extraction
- **Spectral Features**: 232 per frame (log-magnitude spectrum)
- **Temporal Context**: 43 frames (~1 second of audio)
- **Normalization**: Global mean-variance normalization

### Model Inference
- **Input Shape**: [9976] (43 × 232 features)
- **Output Shape**: [3] (3 class probabilities)
- **Quantization**: Int8 with per-tensor scaling

## Performance Characteristics

- **Latency**: ~500ms inference interval
- **Memory Usage**: 
  - Feature buffers: ~40KB
  - Model size: ~50KB
  - Total RAM: ~100KB
- **Throughput**: Real-time audio processing at 44032 Hz

## Integration Points

### ESP-IDF Integration
- **I2S Driver**: Hardware audio capture
- **Heap Caps**: Memory management
- **FreeRTOS**: Task scheduling and synchronization

### TensorFlow Lite Micro
- **Model Format**: .tflite quantized models
- **Operator Support**: Conv2D, Dense, Reshape, etc.
- **Memory Planning**: Static memory allocation

This pipeline demonstrates a complete embedded ML solution with efficient resource usage, modular architecture, and real-time performance capabilities.

## Detailed Function Call Flow

### Main Application Flow
```cpp
// examples/audio_classification/main/main.cpp
app_main()
  → AudioClassificationDemo::run()
    → loadModelFromEmbedded()  // Load TFLite model
    → pipeline->init()         // Initialize inference pipeline
    → while(true) {
        sensor->process()      // Capture audio
        pipeline->infer()      // Run inference
        displayResults()       // Show classification
      }
```

### Inference Pipeline Flow
```cpp
// components/acoustics/api/inference_pipeline.cpp
InferencePipeline::infer(audio_samples)
  → _feature_extractor->process(input_tensor, feature_tensor)
    → ProcessorFeatureExtractor::extractFeatures()
      → STFT processing with KissFFT
      → Log magnitude calculation
      → Global normalization
  → _inference_engine->forward(feature_tensor, output_tensor)
    → TFLiteEngine::forwardCallback()
      → Quantize input: float32 → int8
      → interpreter->Invoke()  // TFLite inference
      → Dequantize output: int8 → float32
```

### Memory Allocation Flow
```cpp
// Feature tensor allocation
InferencePipeline::init()
  → auto feature_spec = _feature_extractor->getOutputSpec()
  → feature_size = feature_spec.first[0]  // 9976 elements
  → _feature_tensor = new Tensor(Float32, {feature_size})
    → Tensor constructor allocates 39904 bytes (9976 * 4)

// Feature extractor buffers
ProcessorFeatureExtractor::allocateBuffers()
  → _fft_input_buffer = heap_caps_malloc(2048 * sizeof(float))
  → _fft_output_buffer = heap_caps_malloc(1025 * sizeof(kiss_fft_cpx))
  → _log_features_buffer = heap_caps_malloc(9976 * sizeof(float))
```

## Code Architectural Features

### 1. Template-Based Type Safety
```cpp
// components/acoustics/core/tensor.hpp
template<typename T>
inline T* dataAs() const {
    return reinterpret_cast<T*>(_data.get());
}
```

### 2. RAII Resource Management
```cpp
// Automatic cleanup in destructors
~ProcessorFeatureExtractor() override {
    deinit();  // Automatically frees all buffers
}
```

### 3. Status-Based Error Handling
```cpp
// components/acoustics/core/status.hpp
#define STATUS_OK() core::Status()
#define STATUS(code, msg) core::Status(code, msg, __FILE__, __LINE__)

// Usage throughout codebase
core::Status process(const Tensor& input, Tensor& output) {
    if (!_initialized) {
        return STATUS(ENXIO, "Feature extractor not initialized");
    }
    // ... processing
    return STATUS_OK();
}
```

### 4. Configuration-Driven Design
```cpp
// Dynamic configuration updates
core::Status updateConfig(const core::ConfigMap& configs) override {
    for (const auto& [key, value] : configs) {
        if (key == "audio_samples") {
            _audio_samples = std::get<int>(value);
            _output_size = _num_frames * _features_per_frame;
        }
    }
}
```

### 5. Hardware Abstraction Layer
```cpp
// components/acoustics/hal/sensor.hpp
class Sensor {
public:
    virtual core::Status process(core::Tensor& output) = 0;
    virtual std::pair<core::Tensor::Shape, core::Tensor::Type> getOutputSpec() const = 0;
};
```

## Quantization Details

### Input Quantization (Float32 → Int8)
```cpp
// components/acoustics/hal/engines/tflite_engine.cpp
float input_scale = input_tensor->params.scale;
int32_t input_zero_point = input_tensor->params.zero_point;

for (size_t i = 0; i < num_elements; ++i) {
    float float_val = input_data[i];
    int32_t quantized_val = static_cast<int32_t>(
        std::round(float_val / input_scale)) + input_zero_point;

    // Clamp to int8 range [-128, 127]
    quantized_val = std::max(-128, std::min(127, quantized_val));
    input_tensor->data.int8[i] = static_cast<int8_t>(quantized_val);
}
```

### Output Dequantization (Int8 → Float32)
```cpp
float output_scale = output_tensor->params.scale;
int32_t output_zero_point = output_tensor->params.zero_point;

for (size_t i = 0; i < num_elements; ++i) {
    int8_t quantized_val = output_tensor->data.int8[i];
    float float_score = static_cast<float>(
        quantized_val - output_zero_point) * output_scale;
    output_data[i] = float_score;
}
```

## Build System Integration

### CMake Configuration
```cmake
# examples/audio_classification/main/CMakeLists.txt
target_add_binary_data(${COMPONENT_LIB}
    "../../../fixed2conv_model.tflite"
    BINARY)
```

### Binary Embedding
```cpp
// Automatic symbol generation by CMake
extern const uint8_t fixed2conv_model_tflite_start[]
    asm("_binary_fixed2conv_model_tflite_start");
extern const uint8_t fixed2conv_model_tflite_end[]
    asm("_binary_fixed2conv_model_tflite_end");
```

This comprehensive pipeline showcases modern C++ embedded development practices with efficient memory management, type safety, and modular architecture suitable for resource-constrained environments.
