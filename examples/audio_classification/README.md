# Audio Classification Example

This example demonstrates how to use the AcousticsLab framework for real-time audio classification using the unified InferencePipeline API.

## Overview

The example showcases:
- **Feature Extraction**: Converting raw audio data to spectral features using STFT
- **Model Inference**: Running TensorFlow Lite Micro models for classification
- **Unified API**: Using the InferencePipeline for end-to-end audio processing
- **Real-time Processing**: Continuous audio capture and classification

## Architecture

```
Audio Input (I2S PDM) → Feature Extraction → TFLite Inference → Classification Results
                           (STFT + Log Mag)     (Neural Network)    (Class Probabilities)
```

### Components Used

1. **Microphone Sensor** (`hal::Sensor`)
   - I2S PDM microphone interface
   - 44.1kHz sampling rate
   - Real-time audio capture

2. **Feature Extractor** (`hal::Processor`)
   - STFT-based spectral analysis
   - Hanning window (2048 samples)
   - Log-magnitude features (232 per frame)

3. **TFLite Engine** (`hal::Engine`)
   - TensorFlow Lite Micro inference
   - Quantized model support
   - Configurable tensor arena

4. **Inference Pipeline** (`api::InferencePipeline`)
   - Unified high-level API
   - Automatic component management
   - Performance monitoring

## Hardware Requirements

- **ESP32-S3** development board
- **I2S PDM microphone** connected to:
  - CLK: GPIO 42
  - DATA: GPIO 41
- **PSRAM** enabled (recommended for better performance)

## Software Requirements

- ESP-IDF v5.0 or later
- AcousticsLab framework components
- TensorFlow Lite Micro component

## Configuration

### Audio Configuration
```cpp
api::InferencePipelineConfig config;
config.audio_samples = 44032;      // ~1 second at 44.1kHz
config.frame_length = 2048;        // STFT frame size
config.hop_length = 1024;          // STFT hop size
config.fft_size = 2048;            // FFT size
config.features_per_frame = 232;   // Features per frame
config.use_psram = true;           // Use PSRAM for buffers
config.normalize_features = true;  // Apply global normalization
```

### Model Configuration
```cpp
config.tensor_arena_size = 2048 * 1024;  // 2MB tensor arena
config.model_name = "audio_classifier";
config.model_version = "1.0.0";
config.labels = {
    {0, "Background Noise"},
    {1, "Speech"},
    {2, "Music"}
};
```

## Usage

### Basic Usage
```cpp
#include "api/inference_pipeline.hpp"

// Create pipeline
api::InferencePipelineConfig config;
auto pipeline = std::make_unique<api::InferencePipeline>(config);

// Initialize
pipeline->init();

// Load model
pipeline->loadModel(model_data, model_size, labels);

// Perform inference
api::InferenceResult result;
pipeline->infer(audio_data, audio_samples, result);

// Get results
auto [top_class, top_score] = result.getTopClass();
std::string label = result.getLabel(top_class);
```

### Advanced Usage with Data Frames
```cpp
// Read from microphone sensor
core::DataFrame<core::Tensor> audio_frame;
microphone->readDataFrame(audio_frame, 44032);

// Perform inference
api::InferenceResult result;
pipeline->inferDataFrame(audio_frame, result);

// Process results
for (size_t i = 0; i < result.scores.size(); ++i) {
    std::string label = result.getLabel(i);
    float score = result.scores[i];
    printf("%s: %.4f\n", label.c_str(), score);
}
```

## Building and Running

1. **Set up ESP-IDF environment**:
   ```bash
   . $IDF_PATH/export.sh
   ```

2. **Configure the project**:
   ```bash
   cd examples/audio_classification
   idf.py set-target esp32s3
   idf.py menuconfig
   ```

3. **Enable PSRAM** (recommended):
   - Component config → ESP32S3-Specific → Support for external, SPI-connected RAM

4. **Build and flash**:
   ```bash
   idf.py build
   idf.py flash monitor
   ```

## Expected Output

```
I (1234) AudioClassification: === Classification Results ===
I (1234) AudioClassification: Frame: 1, Time: 45123 μs
I (1234) AudioClassification:   Background Noise: 0.1234
I (1234) AudioClassification:   Speech: 0.7890
I (1234) AudioClassification:   Music: 0.0876
I (1234) AudioClassification:   >> Top: Speech (0.7890)
I (1234) AudioClassification: =============================

I (1234) AudioClassification: Performance Statistics:
I (1234) AudioClassification:   Total inferences: 10
I (1234) AudioClassification:   Avg feature extraction: 12.34 ms
I (1234) AudioClassification:   Avg inference: 32.10 ms
I (1234) AudioClassification:   Avg total: 45.67 ms
```

## Customization

### Using Your Own Model

1. **Replace model data**:
   ```cpp
   // Include your TFLite model
   #include "your_model.h"
   
   // Use in pipeline
   pipeline->loadModel(your_model_tflite, your_model_tflite_len, your_labels);
   ```

2. **Update labels**:
   ```cpp
   static const std::unordered_map<int, std::string> YOUR_LABELS = {
       {0, "Class1"},
       {1, "Class2"},
       {2, "Class3"}
   };
   ```

### Adjusting Performance

- **Reduce latency**: Decrease `audio_samples` and `hop_length`
- **Improve accuracy**: Increase `frame_length` and `features_per_frame`
- **Save memory**: Reduce `tensor_arena_size` or disable PSRAM

## Troubleshooting

### Common Issues

1. **Memory allocation failures**:
   - Enable PSRAM
   - Reduce tensor arena size
   - Check available heap memory

2. **Audio capture issues**:
   - Verify I2S pin connections
   - Check microphone power supply
   - Ensure correct sampling rate

3. **Inference failures**:
   - Verify model format (TFLite)
   - Check input/output tensor shapes
   - Ensure sufficient tensor arena size

### Debug Tips

- Enable debug logging: `idf.py menuconfig` → Component config → Log output → Default log verbosity → Debug
- Monitor memory usage: `esp_get_free_heap_size()`
- Check performance stats: `pipeline->getPerformanceStats()`

## Performance Characteristics

- **Feature extraction**: ~12-15ms (ESP32-S3 @ 240MHz)
- **Model inference**: ~30-40ms (depends on model complexity)
- **Total latency**: ~45-60ms per inference
- **Memory usage**: ~3-4MB (including PSRAM buffers)

## Next Steps

- Integrate with your own audio classification models
- Implement real-time audio streaming
- Add audio preprocessing (noise reduction, gain control)
- Explore different feature extraction methods
- Optimize for your specific use case
