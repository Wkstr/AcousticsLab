#!/usr/bin/env python3
"""
Improved Model Converter for AcousticsLab
Converts TF.js models to AcousticsLab-compatible format with automatic configuration generation.
"""

import tensorflow as tf
import numpy as np
import os
import glob
import subprocess
import sys
import json
from pathlib import Path

# --- Configuration ---
TFJS_MODEL_JSON = 'tm-my-audio-model/model.json' 
KERAS_H5_FILE = 'tm-my-audio-model/keras_model.h5'
FINAL_TFLITE_FILE = 'tm-my-audio-model/fixed_model.tflite'
REPRESENTATIVE_SAMPLES_DIR = 'real_input_samples/'
METADATA_JSON_FILE = 'tm-my-audio-model/metadata.json' 

# Output files for AcousticsLab
MODEL_DATA_HEADER = 'tm-my-audio-model/model_data.h'
MODEL_CONFIG_HEADER = 'tm-my-audio-model/model_config.h'
ACOUSTICSLAB_CONFIG_HEADER = 'tm-my-audio-model/acousticslab_model.h'

class ModelConverter:
    def __init__(self):
        self.model_info = {}
        self.labels = []
        
    def convert_tfjs_to_keras(self):
        """Convert TF.js model to Keras .h5 format."""
        print(f"--- Step 1: Converting {TFJS_MODEL_JSON} to {KERAS_H5_FILE} ---")
        if not os.path.exists(TFJS_MODEL_JSON):
            print(f"Error: TF.js model file '{TFJS_MODEL_JSON}' not found.")
            sys.exit(1)
            
        command = [
            'tensorflowjs_converter',
            '--input_format=tfjs_layers_model',
            '--output_format=keras',
            TFJS_MODEL_JSON,
            KERAS_H5_FILE
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: tensorflowjs_converter failed.")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            sys.exit(1)
        
        print("✅ TF.js to Keras conversion successful!")

    def load_metadata(self):
        """Load and parse metadata.json file."""
        print(f"--- Step 2: Loading metadata from {METADATA_JSON_FILE} ---")
        if not os.path.exists(METADATA_JSON_FILE):
            print(f"Warning: Metadata file '{METADATA_JSON_FILE}' not found. Using default labels.")
            self.labels = ["Class_0", "Class_1", "Class_2"]  # Default fallback
            return
            
        try:
            with open(METADATA_JSON_FILE, 'r') as f:
                metadata = json.load(f)
            
            if 'wordLabels' in metadata and isinstance(metadata['wordLabels'], list):
                self.labels = metadata['wordLabels']
                print(f"✅ Loaded {len(self.labels)} labels: {self.labels}")
            else:
                raise ValueError("No 'wordLabels' array found in metadata.json")
                
        except Exception as e:
            print(f"Error processing metadata: {e}")
            print("Using default labels as fallback.")
            self.labels = ["Class_0", "Class_1", "Class_2"]

    def create_models(self):
        """Create original and optimized model architectures."""
        print("--- Step 3: Creating model architectures ---")
        
        # Original model for weight loading
        inputs = tf.keras.layers.Input(shape=(43, 232, 1), name='conv2d_1_input')
        x = tf.keras.layers.Conv2D(8, kernel_size=(2, 8), activation='relu', name='conv2d_1')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_1')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_2')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_3')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_3')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_4')(x)
        x_before_flatten = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='max_pooling2d_4')(x)
        x_flat = tf.keras.layers.Flatten(name='flatten_1')(x_before_flatten)
        x_dropout = tf.keras.layers.Dropout(0.25, name='dropout_1')(x_flat) 
        x_dense = tf.keras.layers.Dense(2000, activation='relu', name='dense_1')(x_dropout)
        outputs = tf.keras.layers.Dense(len(self.labels), activation='softmax', name='NewHeadDense')(x_dense)
        self.original_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='original_model')
        
        # Optimized model with Conv2D replacements
        inputs = tf.keras.layers.Input(shape=(43, 232, 1), name='conv2d_1_input')
        x = tf.keras.layers.Conv2D(8, kernel_size=(2, 8), activation='relu', name='conv2d_1')(inputs)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_1')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_2')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_2')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_3')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_3')(x)
        x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation='relu', name='conv2d_4')(x)
        x_before_flatten = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name='max_pooling2d_4')(x)

        # Replace Dense layers with Conv2D
        H, W = x_before_flatten.shape[1], x_before_flatten.shape[2]
        x_conv1 = tf.keras.layers.Conv2D(2000, kernel_size=(H, W), activation='relu', name='dense_1')(x_before_flatten)
        x_flat1 = tf.keras.layers.Flatten(name='flatten_1')(x_conv1)
        x_dropout = tf.keras.layers.Dropout(0.25, name='dropout_1')(x_flat1)
        x_reshaped = tf.keras.layers.Reshape((1, 1, 2000))(x_dropout)
        x_conv2 = tf.keras.layers.Conv2D(len(self.labels), kernel_size=(1, 1), activation='softmax', name='NewHeadDense')(x_reshaped)
        outputs = tf.keras.layers.Flatten(name='output_flatten')(x_conv2)
        
        self.final_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='optimized_model')
        
        print("✅ Model architectures created successfully!")

    def transfer_weights(self):
        """Load weights from .h5 file and transfer to optimized model."""
        print("--- Step 4: Loading and transferring weights ---")
        
        self.original_model.load_weights(KERAS_H5_FILE)
        print(f"✅ Weights loaded from '{KERAS_H5_FILE}'")

        # Transfer weights
        for layer_new in self.final_model.layers:
            if not layer_new.get_weights(): 
                continue
            try:
                layer_orig = self.original_model.get_layer(name=layer_new.name)
                layer_new.set_weights(layer_orig.get_weights())
                print(f"  ✅ Transferred weights: {layer_new.name}")
            except ValueError:
                print(f"  ⚠️  Could not transfer '{layer_new.name}' (structure changed)")

        print("✅ Weight transfer completed!")

    def representative_data_gen(self):
        """Generate representative data for quantization calibration."""
        input_shape = (1, 43, 232, 1) 
        sample_files = glob.glob(os.path.join(REPRESENTATIVE_SAMPLES_DIR, '*.txt'))
        
        if not sample_files:
            print(f"Warning: No sample files found in '{REPRESENTATIVE_SAMPLES_DIR}'. Using random samples.")
            for _ in range(200):
                yield [np.random.rand(*input_shape).astype(np.float32)]
            return
        
        num_samples = min(len(sample_files), 200)
        print(f"Using {num_samples} samples for quantization calibration...")
        
        for i in range(num_samples):
            try:
                sample_data = np.loadtxt(sample_files[i], delimiter=',')
                model_input = sample_data.reshape(input_shape).astype(np.float32)
                yield [model_input]
            except Exception as e:
                print(f"Warning: Could not process sample '{sample_files[i]}': {e}")
                continue

    def quantize_and_convert(self):
        """Quantize model and convert to TFLite format."""
        print("--- Step 5: Quantizing and converting to TFLite ---")
        
        @tf.function
        def model_func(input_tensor):
            return self.final_model(input_tensor)

        fixed_input_spec = tf.TensorSpec(shape=[1, 43, 232, 1], dtype=tf.float32)
        concrete_func = model_func.get_concrete_function(fixed_input_spec)

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = self.representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        self.tflite_model = converter.convert()

        with open(FINAL_TFLITE_FILE, 'wb') as f:
            f.write(self.tflite_model)
        
        print(f"✅ Quantized TFLite model saved: {FINAL_TFLITE_FILE}")
        
        # Extract model information
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        
        self.model_info = {
            'input_shape': input_details['shape'].tolist(),
            'input_dtype': str(input_details['dtype']),
            'input_scale': float(input_details['quantization'][0]),
            'input_zero_point': int(input_details['quantization'][1]),
            'output_shape': output_details['shape'].tolist(),
            'output_dtype': str(output_details['dtype']),
            'output_scale': float(output_details['quantization'][0]),
            'output_zero_point': int(output_details['quantization'][1]),
            'model_size': len(self.tflite_model),
            'num_classes': len(self.labels),
            'class_labels': self.labels
        }
        
        print(f"✅ Model info extracted: {self.model_info['input_shape']} -> {self.model_info['output_shape']}")

    def generate_headers(self):
        """Generate all necessary header files for AcousticsLab."""
        print("--- Step 6: Generating header files ---")
        
        # Generate model data header
        self._generate_model_data_header()
        
        # Generate model configuration header
        self._generate_model_config_header()
        
        # Generate AcousticsLab integration header
        self._generate_acousticslab_header()
        
        print("✅ All header files generated successfully!")

    def _generate_model_data_header(self):
        """Generate C++ header with model binary data."""
        var_name = "g_model_data"
        
        content = f'''#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

// Auto-generated model data for AcousticsLab
// Model size: {len(self.tflite_model)} bytes
// Classes: {len(self.labels)}

static constexpr unsigned int {var_name}_len = {len(self.tflite_model)};

alignas(16) static const unsigned char {var_name}[] = {{
'''
        
        # Add binary data
        for i, byte in enumerate(self.tflite_model):
            if i % 12 == 0:
                content += '\n  '
            content += f'0x{byte:02x},'
            if i % 12 != 11:
                content += ' '
        
        content = content.rstrip(', ') + '\n};\n\n#endif // MODEL_DATA_H_\n'
        
        with open(MODEL_DATA_HEADER, 'w') as f:
            f.write(content)
        
        print(f"✅ Model data header: {MODEL_DATA_HEADER}")

    def _generate_model_config_header(self):
        """Generate C++ header with model configuration."""
        content = f'''#ifndef MODEL_CONFIG_H_
#define MODEL_CONFIG_H_

// Auto-generated model configuration for AcousticsLab

// Model specifications
static constexpr int kModelInputHeight = {self.model_info['input_shape'][1]};
static constexpr int kModelInputWidth = {self.model_info['input_shape'][2]};
static constexpr int kModelInputChannels = {self.model_info['input_shape'][3]};
static constexpr int kModelInputSize = {np.prod(self.model_info['input_shape'][1:])};

static constexpr int kModelOutputSize = {self.model_info['output_shape'][1]};
static constexpr int kNumClasses = {self.model_info['num_classes']};

// Quantization parameters
static constexpr float kInputScale = {self.model_info['input_scale']}f;
static constexpr int kInputZeroPoint = {self.model_info['input_zero_point']};
static constexpr float kOutputScale = {self.model_info['output_scale']}f;
static constexpr int kOutputZeroPoint = {self.model_info['output_zero_point']};

// Class labels
static const char* kClassLabels[kNumClasses] = {{
'''
        
        for label in self.labels:
            content += f'    "{label}",\n'
        
        content += '''};

#endif // MODEL_CONFIG_H_
'''
        
        with open(MODEL_CONFIG_HEADER, 'w') as f:
            f.write(content)
        
        print(f"✅ Model config header: {MODEL_CONFIG_HEADER}")

    def _generate_acousticslab_header(self):
        """Generate unified header for easy AcousticsLab integration."""
        content = f'''#ifndef ACOUSTICSLAB_MODEL_H_
#define ACOUSTICSLAB_MODEL_H_

// Auto-generated unified header for AcousticsLab integration
// Include this file in your AcousticsLab project for automatic model loading

#include "model_data.h"
#include "model_config.h"

// Convenience functions for AcousticsLab integration
namespace acousticslab {{

inline const unsigned char* getModelData() {{
    return g_model_data;
}}

inline unsigned int getModelSize() {{
    return g_model_data_len;
}}

inline int getNumClasses() {{
    return kNumClasses;
}}

inline const char* getClassName(int class_id) {{
    if (class_id >= 0 && class_id < kNumClasses) {{
        return kClassLabels[class_id];
    }}
    return "Unknown";
}}

inline int getModelInputSize() {{
    return kModelInputSize;
}}

inline float getInputScale() {{
    return kInputScale;
}}

inline int getInputZeroPoint() {{
    return kInputZeroPoint;
}}

inline float getOutputScale() {{
    return kOutputScale;
}}

inline int getOutputZeroPoint() {{
    return kOutputZeroPoint;
}}

}} // namespace acousticslab

#endif // ACOUSTICSLAB_MODEL_H_
'''
        
        with open(ACOUSTICSLAB_CONFIG_HEADER, 'w') as f:
            f.write(content)
        
        print(f"✅ AcousticsLab integration header: {ACOUSTICSLAB_CONFIG_HEADER}")

    def run(self):
        """Execute the complete conversion pipeline."""
        print("🚀 Starting AcousticsLab Model Conversion Pipeline")
        print("=" * 60)
        
        self.convert_tfjs_to_keras()
        self.load_metadata()
        self.create_models()
        self.transfer_weights()
        self.quantize_and_convert()
        self.generate_headers()
        
        print("\n" + "=" * 60)
        print("🎉 Conversion completed successfully!")
        print(f"📁 Generated files:")
        print(f"   - TFLite model: {FINAL_TFLITE_FILE}")
        print(f"   - Model data: {MODEL_DATA_HEADER}")
        print(f"   - Model config: {MODEL_CONFIG_HEADER}")
        print(f"   - AcousticsLab integration: {ACOUSTICSLAB_CONFIG_HEADER}")
        print(f"\n📋 Model summary:")
        print(f"   - Input shape: {self.model_info['input_shape']}")
        print(f"   - Output classes: {self.model_info['num_classes']}")
        print(f"   - Model size: {self.model_info['model_size']:,} bytes")
        print(f"   - Labels: {', '.join(self.labels)}")
        print(f"\n✨ Ready for AcousticsLab integration!")

if __name__ == '__main__':
    converter = ModelConverter()
    converter.run()
