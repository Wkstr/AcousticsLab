#!/usr/bin/env python3
"""
Improved Model Converter for AcousticsLab
Converts TF.js models to AcousticsLab-compatible format with automatic configuration generation.
"""

# Standard library imports
import argparse
import glob
import json
import logging
import os
import subprocess
import sys

# Third-party imports
import numpy as np
import tensorflow as tf

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert TF.js models to AcousticsLab-compatible format"
    )

    parser.add_argument(
        "--tfjs-model", required=True, help="Path to TF.js model.json file"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for generated files (default: ./output)",
    )
    parser.add_argument(
        "--samples-dir",
        help="Directory containing representative samples for quantization",
    )
    parser.add_argument(
        "--metadata", help="Path to metadata.json file containing class labels"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser.parse_args()


class ModelConverter:
    def __init__(self, config):
        self.config = config
        self.model_info = {}
        self.labels = []

        # Set up file paths based on config
        self.tfjs_model_json = config.tfjs_model
        self.output_dir = config.output_dir
        self.samples_dir = config.samples_dir
        self.metadata_file = config.metadata

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Derived file paths
        model_name = os.path.splitext(os.path.basename(self.tfjs_model_json))[0]
        self.keras_h5_file = os.path.join(self.output_dir, f"{model_name}.h5")
        self.tflite_file = os.path.join(self.output_dir, f"{model_name}.tflite")
        self.model_data_header = os.path.join(self.output_dir, "model_data.h")
        self.model_config_header = os.path.join(self.output_dir, "model_config.h")
        self.acousticslab_header = os.path.join(self.output_dir, "acousticslab_model.h")

    def convert_tfjs_to_keras(self):
        """Convert TF.js model to Keras .h5 format."""
        logger.info(
            "Step 1: Converting %s to %s", self.tfjs_model_json, self.keras_h5_file
        )
        if not os.path.exists(self.tfjs_model_json):
            logger.error("TF.js model file '%s' not found.", self.tfjs_model_json)
            sys.exit(1)

        command = [
            "tensorflowjs_converter",
            "--input_format=tfjs_layers_model",
            "--output_format=keras",
            self.tfjs_model_json,
            self.keras_h5_file,
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("tensorflowjs_converter failed.")
            logger.error("STDOUT: %s", result.stdout)
            logger.error("STDERR: %s", result.stderr)
            sys.exit(1)

        logger.info("✅ TF.js to Keras conversion successful!")

    def load_metadata(self):
        """Load and parse metadata.json file."""
        if not self.metadata_file:
            logger.info("Step 2: No metadata file specified, using default labels")
            self.labels = ["Class_0", "Class_1", "Class_2"]  # Default fallback
            return

        logger.info("Step 2: Loading metadata from %s", self.metadata_file)
        if not os.path.exists(self.metadata_file):
            logger.warning(
                "Metadata file '%s' not found. Using default labels.",
                self.metadata_file,
            )
            self.labels = ["Class_0", "Class_1", "Class_2"]  # Default fallback
            return

        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)

            if "wordLabels" in metadata and isinstance(metadata["wordLabels"], list):
                self.labels = metadata["wordLabels"]
                logger.info("✅ Loaded %d labels: %s", len(self.labels), self.labels)
            else:
                raise ValueError("No 'wordLabels' array found in metadata.json")

        except Exception as e:
            logger.error("Error processing metadata: %s", e)
            logger.info("Using default labels as fallback.")
            self.labels = ["Class_0", "Class_1", "Class_2"]

    def create_models(self):
        """Create original and optimized model architectures."""
        logger.info("Step 3: Creating model architectures")

        # Original model for weight loading
        inputs = tf.keras.layers.Input(shape=(43, 232, 1), name="conv2d_1_input")
        x = tf.keras.layers.Conv2D(
            8, kernel_size=(2, 8), activation="relu", name="conv2d_1"
        )(inputs)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_1"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_2"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_2"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_3"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_3"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_4"
        )(x)
        x_before_flatten = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(1, 2), name="max_pooling2d_4"
        )(x)
        x_flat = tf.keras.layers.Flatten(name="flatten_1")(x_before_flatten)
        x_dropout = tf.keras.layers.Dropout(0.25, name="dropout_1")(x_flat)
        x_dense = tf.keras.layers.Dense(2000, activation="relu", name="dense_1")(
            x_dropout
        )
        outputs = tf.keras.layers.Dense(
            len(self.labels), activation="softmax", name="NewHeadDense"
        )(x_dense)
        self.original_model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="original_model"
        )

        # Optimized model with Conv2D replacements
        inputs = tf.keras.layers.Input(shape=(43, 232, 1), name="conv2d_1_input")
        x = tf.keras.layers.Conv2D(
            8, kernel_size=(2, 8), activation="relu", name="conv2d_1"
        )(inputs)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_1"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_2"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_2"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_3"
        )(x)
        x = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_3"
        )(x)
        x = tf.keras.layers.Conv2D(
            32, kernel_size=(2, 4), activation="relu", name="conv2d_4"
        )(x)
        x_before_flatten = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(1, 2), name="max_pooling2d_4"
        )(x)

        # Replace Dense layers with Conv2D
        H, W = x_before_flatten.shape[1], x_before_flatten.shape[2]
        x_conv1 = tf.keras.layers.Conv2D(
            2000, kernel_size=(H, W), activation="relu", name="dense_1"
        )(x_before_flatten)
        x_flat1 = tf.keras.layers.Flatten(name="flatten_1")(x_conv1)
        x_dropout = tf.keras.layers.Dropout(0.25, name="dropout_1")(x_flat1)
        x_reshaped = tf.keras.layers.Reshape((1, 1, 2000))(x_dropout)
        x_conv2 = tf.keras.layers.Conv2D(
            len(self.labels),
            kernel_size=(1, 1),
            activation="softmax",
            name="NewHeadDense",
        )(x_reshaped)
        outputs = tf.keras.layers.Flatten(name="output_flatten")(x_conv2)

        self.final_model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="optimized_model"
        )

        logger.info("✅ Model architectures created successfully!")

    def transfer_weights(self):
        """Load weights from .h5 file and transfer to optimized model."""
        logger.info("Step 4: Loading and transferring weights")

        self.original_model.load_weights(self.keras_h5_file)
        logger.info("✅ Weights loaded from '%s'", self.keras_h5_file)

        # Transfer weights
        for layer_new in self.final_model.layers:
            if not layer_new.get_weights():
                continue
            try:
                layer_orig = self.original_model.get_layer(name=layer_new.name)
                layer_new.set_weights(layer_orig.get_weights())
                logger.debug("  ✅ Transferred weights: %s", layer_new.name)
            except ValueError:
                logger.warning(
                    "  ⚠️  Could not transfer '%s' (structure changed)", layer_new.name
                )

        logger.info("✅ Weight transfer completed!")

    def representative_data_gen(self):
        """Generate representative data for quantization calibration."""
        input_shape = (1, 43, 232, 1)

        if not self.samples_dir:
            logger.info(
                "No samples directory specified, using random samples for quantization"
            )
            for _ in range(200):
                yield [np.random.rand(*input_shape).astype(np.float32)]
            return

        sample_files = glob.glob(os.path.join(self.samples_dir, "*.txt"))

        if not sample_files:
            logger.warning(
                "No sample files found in '%s'. Using random samples.", self.samples_dir
            )
            for _ in range(200):
                yield [np.random.rand(*input_shape).astype(np.float32)]
            return

        num_samples = min(len(sample_files), 200)
        logger.info("Using %d samples for quantization calibration...", num_samples)

        for i in range(num_samples):
            try:
                sample_data = np.loadtxt(sample_files[i], delimiter=",")
                model_input = sample_data.reshape(input_shape).astype(np.float32)
                yield [model_input]
            except Exception as e:
                logger.warning("Could not process sample '%s': %s", sample_files[i], e)
                continue

    def quantize_and_convert(self):
        """Quantize model and convert to TFLite format."""
        logger.info("Step 5: Quantizing and converting to TFLite")

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

        with open(self.tflite_file, "wb") as f:
            f.write(self.tflite_model)

        logger.info("✅ Quantized TFLite model saved: %s", self.tflite_file)

        # Extract model information
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        self.model_info = {
            "input_shape": input_details["shape"].tolist(),
            "input_dtype": str(input_details["dtype"]),
            "input_scale": float(input_details["quantization"][0]),
            "input_zero_point": int(input_details["quantization"][1]),
            "output_shape": output_details["shape"].tolist(),
            "output_dtype": str(output_details["dtype"]),
            "output_scale": float(output_details["quantization"][0]),
            "output_zero_point": int(output_details["quantization"][1]),
            "model_size": len(self.tflite_model),
            "num_classes": len(self.labels),
            "class_labels": self.labels,
        }

        logger.info(
            "✅ Model info extracted: %s -> %s",
            self.model_info["input_shape"],
            self.model_info["output_shape"],
        )

    def generate_headers(self):
        """Generate all necessary header files for AcousticsLab."""
        logger.info("Step 6: Generating header files")

        # Generate model data header
        self._generate_model_data_header()

        # Generate model configuration header
        self._generate_model_config_header()

        # Generate AcousticsLab integration header
        self._generate_acousticslab_header()

        logger.info("✅ All header files generated successfully!")

    def _generate_model_data_header(self):
        """Generate C++ header with model binary data."""
        var_name = "g_model_data"

        content = f"""#ifndef MODEL_DATA_H_
#define MODEL_DATA_H_

// Auto-generated model data for AcousticsLab
// Model size: {len(self.tflite_model)} bytes
// Classes: {len(self.labels)}

static constexpr unsigned int {var_name}_len = {len(self.tflite_model)};

alignas(16) static const unsigned char {var_name}[] = {{
"""

        # Add binary data
        for i, byte in enumerate(self.tflite_model):
            if i % 12 == 0:
                content += "\n  "
            content += f"0x{byte:02x},"
            if i % 12 != 11:
                content += " "

        content = content.rstrip(", ") + "\n};\n\n#endif // MODEL_DATA_H_\n"

        with open(self.model_data_header, "w") as f:
            f.write(content)

        logger.info("✅ Model data header: %s", self.model_data_header)

    def _generate_model_config_header(self):
        """Generate C++ header with model configuration."""
        content = f"""#ifndef MODEL_CONFIG_H_
#define MODEL_CONFIG_H_

// Auto-generated model configuration for AcousticsLab

// Model specifications
static constexpr int kModelInputHeight = {self.model_info["input_shape"][1]};
static constexpr int kModelInputWidth = {self.model_info["input_shape"][2]};
static constexpr int kModelInputChannels = {self.model_info["input_shape"][3]};
static constexpr int kModelInputSize = {np.prod(self.model_info["input_shape"][1:])};

static constexpr int kModelOutputSize = {self.model_info["output_shape"][1]};
static constexpr int kNumClasses = {self.model_info["num_classes"]};

// Quantization parameters
static constexpr float kInputScale = {self.model_info["input_scale"]}f;
static constexpr int kInputZeroPoint = {self.model_info["input_zero_point"]};
static constexpr float kOutputScale = {self.model_info["output_scale"]}f;
static constexpr int kOutputZeroPoint = {self.model_info["output_zero_point"]};

// Class labels
static const char* kClassLabels[kNumClasses] = {{
"""

        for label in self.labels:
            content += f'    "{label}",\n'

        content += """};

#endif // MODEL_CONFIG_H_
"""

        with open(self.model_config_header, "w") as f:
            f.write(content)

        logger.info("✅ Model config header: %s", self.model_config_header)

    def _generate_acousticslab_header(self):
        """Generate unified header for easy AcousticsLab integration."""
        content = """#ifndef ACOUSTICSLAB_MODEL_H_
#define ACOUSTICSLAB_MODEL_H_

// Auto-generated unified header for AcousticsLab integration
// Include this file in your AcousticsLab project for automatic model loading

#include "model_data.h"
#include "model_config.h"

// Convenience functions for AcousticsLab integration
namespace acousticslab {

inline const unsigned char* getModelData() {
    return g_model_data;
}

inline unsigned int getModelSize() {
    return g_model_data_len;
}

inline int getNumClasses() {
    return kNumClasses;
}

inline const char* getClassName(int class_id) {
    if (class_id >= 0 && class_id < kNumClasses) {
        return kClassLabels[class_id];
    }
    return "Unknown";
}

inline int getModelInputSize() {
    return kModelInputSize;
}

inline float getInputScale() {
    return kInputScale;
}

inline int getInputZeroPoint() {
    return kInputZeroPoint;
}

inline float getOutputScale() {
    return kOutputScale;
}

inline int getOutputZeroPoint() {
    return kOutputZeroPoint;
}

} // namespace acousticslab

#endif // ACOUSTICSLAB_MODEL_H_
"""

        with open(self.acousticslab_header, "w") as f:
            f.write(content)

        logger.info("✅ AcousticsLab integration header: %s", self.acousticslab_header)

    def run(self):
        """Execute the complete conversion pipeline."""
        logger.info("🚀 Starting AcousticsLab Model Conversion Pipeline")
        logger.info("=" * 60)

        self.convert_tfjs_to_keras()
        self.load_metadata()
        self.create_models()
        self.transfer_weights()
        self.quantize_and_convert()
        self.generate_headers()

        logger.info("\n" + "=" * 60)
        logger.info("🎉 Conversion completed successfully!")
        logger.info("📁 Generated files:")
        logger.info("   - TFLite model: %s", self.tflite_file)
        logger.info("   - Model data: %s", self.model_data_header)
        logger.info("   - Model config: %s", self.model_config_header)
        logger.info("   - AcousticsLab integration: %s", self.acousticslab_header)
        logger.info("\n📋 Model summary:")
        logger.info("   - Input shape: %s", self.model_info["input_shape"])
        logger.info("   - Output classes: %s", self.model_info["num_classes"])
        logger.info("   - Model size: %s bytes", f"{self.model_info['model_size']:,}")
        logger.info("   - Labels: %s", ", ".join(self.labels))
        logger.info("\n✨ Ready for AcousticsLab integration!")


if __name__ == "__main__":
    args = parse_arguments()

    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    converter = ModelConverter(args)
    converter.run()
