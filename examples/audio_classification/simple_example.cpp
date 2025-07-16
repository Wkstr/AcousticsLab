/**
 * @file simple_example.cpp
 * @brief Simple audio classification example using AcousticsLab InferencePipeline
 * 
 * This example demonstrates the minimal code needed to perform audio classification
 * using the AcousticsLab framework.
 */

#include "api/context.hpp"
#include "api/inference_pipeline.hpp"
#include "hal/sensor.hpp"
#include <memory>

// Example: Classify audio into 3 categories
static const std::unordered_map<int, std::string> LABELS = {
    {0, "Silence"},
    {1, "Speech"}, 
    {2, "Music"}
};

// Placeholder model data (replace with your actual TFLite model)
extern const unsigned char my_model_tflite[];
extern const size_t my_model_tflite_len;

void simple_audio_classification_example()
{
    // Step 1: Initialize AcousticsLab context
    api::Context::init();
    
    // Step 2: Create inference pipeline with default configuration
    api::InferencePipelineConfig config;
    config.labels = LABELS;
    
    auto pipeline = std::make_unique<api::InferencePipeline>(config);
    
    // Step 3: Initialize pipeline
    if (!pipeline->init()) {
        printf("Failed to initialize pipeline\n");
        return;
    }
    
    // Step 4: Load your TFLite model
    if (!pipeline->loadModel(my_model_tflite, my_model_tflite_len, LABELS)) {
        printf("Failed to load model\n");
        return;
    }
    
    // Step 5: Get microphone sensor
    auto* microphone = hal::SensorRegistry::getSensor(2); // Microphone sensor ID
    if (!microphone || !microphone->init()) {
        printf("Failed to initialize microphone\n");
        return;
    }
    
    // Step 6: Perform classification loop
    core::DataFrame<core::Tensor> audio_frame;
    api::InferenceResult result;
    
    while (true) {
        // Read audio data (44032 samples ≈ 1 second at 44.1kHz)
        if (microphone->readDataFrame(audio_frame, 44032)) {
            
            // Perform inference
            if (pipeline->inferDataFrame(audio_frame, result)) {
                
                // Get top prediction
                auto [class_id, confidence] = result.getTopClass();
                std::string label = result.getLabel(class_id);
                
                printf("Detected: %s (%.2f%%)\n", label.c_str(), confidence * 100);
                
                // Print all scores
                for (size_t i = 0; i < result.scores.size(); ++i) {
                    printf("  %s: %.3f\n", result.getLabel(i).c_str(), result.scores[i]);
                }
            }
        }
        
        // Wait 1 second before next classification
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

/**
 * Alternative: Process pre-recorded audio data
 */
void classify_audio_buffer(const float* audio_data, size_t num_samples)
{
    // Initialize pipeline
    api::InferencePipelineConfig config;
    config.labels = LABELS;
    auto pipeline = std::make_unique<api::InferencePipeline>(config);
    
    pipeline->init();
    pipeline->loadModel(my_model_tflite, my_model_tflite_len, LABELS);
    
    // Perform inference on buffer
    api::InferenceResult result;
    if (pipeline->infer(audio_data, num_samples, result)) {
        
        auto [class_id, confidence] = result.getTopClass();
        printf("Classification: %s (%.2f%%)\n", 
               result.getLabel(class_id).c_str(), confidence * 100);
    }
}

/**
 * Advanced: Custom configuration
 */
void advanced_classification_example()
{
    // Custom configuration for better performance or accuracy
    api::InferencePipelineConfig config;
    
    // Audio processing settings
    config.audio_samples = 44032;        // 1 second of audio
    config.frame_length = 2048;          // STFT frame size
    config.hop_length = 1024;            // STFT hop size
    config.features_per_frame = 232;     // Features per frame
    config.normalize_features = true;    // Apply normalization
    
    // Memory settings
    config.use_psram = true;             // Use PSRAM for large buffers
    config.tensor_arena_size = 2048 * 1024; // 2MB for TFLite
    
    // Model settings
    config.model_name = "my_audio_classifier";
    config.model_version = "2.0.0";
    config.labels = LABELS;
    
    auto pipeline = std::make_unique<api::InferencePipeline>(config);
    
    // Initialize and use pipeline...
    pipeline->init();
    // ... rest of the code
}

/**
 * Performance monitoring example
 */
void performance_monitoring_example()
{
    api::InferencePipelineConfig config;
    config.labels = LABELS;
    auto pipeline = std::make_unique<api::InferencePipeline>(config);
    
    pipeline->init();
    pipeline->loadModel(my_model_tflite, my_model_tflite_len, LABELS);
    
    // Run several inferences
    auto* microphone = hal::SensorRegistry::getSensor(2);
    microphone->init();
    
    core::DataFrame<core::Tensor> audio_frame;
    api::InferenceResult result;
    
    for (int i = 0; i < 10; ++i) {
        if (microphone->readDataFrame(audio_frame, 44032)) {
            pipeline->inferDataFrame(audio_frame, result);
        }
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
    // Get performance statistics
    auto stats = pipeline->getPerformanceStats();
    printf("Performance Statistics:\n");
    printf("  Total inferences: %zu\n", stats.total_inferences);
    printf("  Avg feature extraction: %.2f ms\n", stats.avg_feature_extraction_ms);
    printf("  Avg inference: %.2f ms\n", stats.avg_inference_ms);
    printf("  Avg total time: %.2f ms\n", stats.avg_total_ms);
}
