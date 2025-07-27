#!/usr/bin/env python3
"""Optimized Model Converter for AcousticsLab. Converts TF.js models to TFLite."""

import tensorflow as tf
import numpy as np
import logging
import argparse
import os
import glob
import sys
import json
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any, Iterator

from scipy.io import wavfile
from scipy.signal import resample
from tensorflow.lite.python.interpreter import Interpreter
from tensorflowjs.converters.converter import (
    dispatch_tensorflowjs_to_keras_h5_conversion,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

DEFAULT_TFJS_MODEL_PATH = _PROJECT_ROOT / "tm-my-audio-model" / "model.json"
DEFAULT_METADATA_PATH = _PROJECT_ROOT / "tm-my-audio-model" / "metadata.json"
DEFAULT_KERAS_OUTPUT_PATH = _PROJECT_ROOT / "tm-my-audio-model" / "converted_model.h5"
DEFAULT_TFLITE_OUTPUT_PATH = (
    _PROJECT_ROOT / "tm-my-audio-model" / "quantized_model.tflite"
)
DEFAULT_WAV_INPUT_DIR = _PROJECT_ROOT / "datasound"
DEFAULT_PREPROCESSING_MODEL_PATH = _PROJECT_ROOT / "models" / "preprocessing.tflite"


class ConversionState(Enum):
    INITIALIZED = "initialized"
    TFJS_CONVERTED = "tfjs_converted"
    METADATA_LOADED = "metadata_loaded"
    MODELS_CREATED = "models_created"
    WEIGHTS_TRANSFERRED = "weights_transferred"
    QUANTIZED = "quantized"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelConverter:
    def __init__(
        self,
        tfjs_model_path: str = str(DEFAULT_TFJS_MODEL_PATH),
        metadata_path: str = str(DEFAULT_METADATA_PATH),
        keras_output_path: str = str(DEFAULT_KERAS_OUTPUT_PATH),
        tflite_output_path: str = str(DEFAULT_TFLITE_OUTPUT_PATH),
        wav_input_dir: str = str(DEFAULT_WAV_INPUT_DIR),
        preprocessing_model_path: str = str(DEFAULT_PREPROCESSING_MODEL_PATH),
    ):
        paths = {
            "tfjs_model_path": tfjs_model_path,
            "metadata_path": metadata_path,
            "keras_output_path": keras_output_path,
            "tflite_output_path": tflite_output_path,
            "wav_input_dir": wav_input_dir,
            "preprocessing_model_path": preprocessing_model_path,
        }

        for param_name, path in paths.items():
            if not isinstance(path, str):
                raise TypeError(f"{param_name} must be a string, got {type(path)}")
            if not path or not path.strip():
                raise ValueError(f"{param_name} cannot be empty or None")

        # Public file path attributes (assigned early for easy access)
        self.tfjs_model_path = tfjs_model_path.strip()
        self.metadata_path = metadata_path.strip()
        self.keras_output_path = keras_output_path.strip()
        self.tflite_output_path = tflite_output_path.strip()
        self.wav_input_dir = wav_input_dir.strip()
        self.preprocessing_model_path = preprocessing_model_path.strip()

        # Private attributes for internal state management
        self._model_info: Dict[str, Any] = {}
        self._labels: List[str] = []
        self._conversion_state: ConversionState = ConversionState.INITIALIZED
        self._original_model: Optional[tf.keras.Model] = None
        self._final_model: Optional[tf.keras.Model] = None
        self._tflite_model: Optional[bytes] = None

        logging.info(
            f"🔧 ModelConverter initialized with state: {self._conversion_state.value}"
        )

    def _validate_state(self, expected_state: ConversionState) -> None:
        if self._conversion_state != expected_state:
            error_msg = f"Invalid operation: expected state '{expected_state.value}', but current state is '{self._conversion_state.value}'"

            if self._conversion_state == ConversionState.FAILED:
                error_msg += "\nThe conversion process has failed. Please create a new ModelConverter instance."
            elif self._conversion_state == ConversionState.COMPLETED:
                error_msg += (
                    "\nThe conversion process has already completed successfully."
                )
            else:
                state_order = [
                    ConversionState.INITIALIZED,
                    ConversionState.TFJS_CONVERTED,
                    ConversionState.METADATA_LOADED,
                    ConversionState.MODELS_CREATED,
                    ConversionState.WEIGHTS_TRANSFERRED,
                    ConversionState.QUANTIZED,
                    ConversionState.COMPLETED,
                ]

                try:
                    current_idx = state_order.index(self._conversion_state)
                    if current_idx < len(state_order) - 1:
                        next_state = state_order[current_idx + 1]
                        error_msg += f"\nNext expected operation corresponds to state: '{next_state.value}'"
                except ValueError:
                    pass

            raise RuntimeError(error_msg)

    def _set_state(self, new_state: ConversionState) -> None:
        logging.info(
            f"🔄 State transition: {self._conversion_state.value} → {new_state.value}"
        )
        self._conversion_state = new_state

    def _generate_features_from_wavs(self) -> Iterator[np.ndarray]:
        logging.info(
            "--- Starting feature generation from WAV files for quantization ---"
        )

        if not os.path.exists(self.wav_input_dir):
            raise FileNotFoundError(
                f"WAV input directory not found: '{self.wav_input_dir}'"
            )

        wav_files = glob.glob(os.path.join(self.wav_input_dir, "*.wav"))
        wav_files += glob.glob(os.path.join(self.wav_input_dir, "*.WAV"))

        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in '{self.wav_input_dir}'")

        logging.info(
            f"Found {len(wav_files)} WAV files. Processing them to generate representative data..."
        )

        interpreter = Interpreter(model_path=self.preprocessing_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        target_len = input_details["shape"][1]
        target_sr = 44100

        for wav_path in wav_files:
            try:
                sampling_rate, data = wavfile.read(wav_path)
                data = data.astype(np.float32) / 32768.0
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                if sampling_rate != target_sr:
                    data = resample(data, int(len(data) * (target_sr / sampling_rate)))
                if len(data) > target_len:
                    data = data[
                        (len(data) - target_len) // 2 : (len(data) + target_len) // 2
                    ]
                else:
                    data = np.pad(data, (0, target_len - len(data)), "constant")

                model_input_audio = data.reshape(input_details["shape"])

                interpreter.set_tensor(input_details["index"], model_input_audio)
                interpreter.invoke()
                feature_map = interpreter.get_tensor(output_details["index"])

                yield [feature_map.astype(np.float32)]

            except Exception as e:
                logging.warning(
                    f"Could not process WAV file '{os.path.basename(wav_path)}': {e}. Skipping."
                )
                continue

    def _representative_data_gen(self):
        try:
            yield from self._generate_features_from_wavs()
        except FileNotFoundError:
            logging.error(
                "Failed to generate data from WAVs. Falling back to 200 random samples for calibration."
            )

    def convert_tfjs_to_keras(self) -> None:
        self._validate_state(ConversionState.INITIALIZED)

        logging.info(
            f"--- Step 1: Converting {self.tfjs_model_path} to {self.keras_output_path} ---"
        )

        # Validate input file exists and is readable
        if not os.path.exists(self.tfjs_model_path):
            self._set_state(ConversionState.FAILED)
            raise FileNotFoundError(
                f"TF.js model file not found: '{self.tfjs_model_path}'\n"
                f"Please ensure the file exists and the path is correct."
            )

        if not os.access(self.tfjs_model_path, os.R_OK):
            self._set_state(ConversionState.FAILED)
            raise PermissionError(
                f"TF.js model file is not readable: '{self.tfjs_model_path}'\n"
                f"Please check file permissions."
            )

        # Ensure output directory exists and is writable
        output_dir = os.path.dirname(self.keras_output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError:
                self._set_state(ConversionState.FAILED)
                raise PermissionError(
                    f"Cannot create output directory: '{output_dir}'\n"
                    f"Please check directory permissions."
                )

        try:
            logging.info("🔄 Converting TF.js model using tensorflowjs converter...")

            dispatch_tensorflowjs_to_keras_h5_conversion(
                self.tfjs_model_path,
                self.keras_output_path,
            )

            logging.info("TensorFlow.js to Keras conversion completed successfully")

            # Verify the output file was created successfully
            if not os.path.exists(self.keras_output_path):
                self._set_state(ConversionState.FAILED)
                raise RuntimeError(
                    f"Keras model file was not created: '{self.keras_output_path}'\n"
                    f"The conversion process completed but no output file was generated."
                )

            # Verify the output file is not empty
            if os.path.getsize(self.keras_output_path) == 0:
                self._set_state(ConversionState.FAILED)
                raise RuntimeError(
                    f"Keras model file is empty: '{self.keras_output_path}'\n"
                    f"The conversion process may have failed silently."
                )

            self._set_state(ConversionState.TFJS_CONVERTED)
            file_size = os.path.getsize(self.keras_output_path)
            logging.info(
                f"✅ TF.js to Keras conversion successful! Output size: {file_size:,} bytes"
            )

        except Exception as e:
            self._set_state(ConversionState.FAILED)
            # Check if it's a TensorFlow.js specific error based on the error message
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in ["tensorflowjs", "tfjs", "model.json", "invalid model"]
            ):
                raise RuntimeError(
                    f"TensorFlow.js conversion failed: {str(e)}\n"
                    f"Please ensure the input file is a valid TF.js model."
                ) from e
            else:
                raise RuntimeError(
                    f"Unexpected error during TF.js to Keras conversion: {str(e)}\n"
                    f"Please check the input file format and try again."
                ) from e

    def load_metadata(self) -> None:
        self._validate_state(ConversionState.TFJS_CONVERTED)

        logging.info(f"--- Step 2: Loading metadata from {self.metadata_path} ---")

        if not os.path.exists(self.metadata_path):
            self._set_state(ConversionState.FAILED)
            raise FileNotFoundError(
                f"Metadata file not found: '{self.metadata_path}'\n"
                f"This file is required and must contain class labels in 'wordLabels' key."
            )

        if not os.access(self.metadata_path, os.R_OK):
            self._set_state(ConversionState.FAILED)
            raise PermissionError(
                f"Metadata file is not readable: '{self.metadata_path}'\n"
                f"Please check file permissions."
            )

        try:
            logging.info(f"🔄 Reading metadata from {self.metadata_path}...")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Metadata file must contain a JSON object, got {type(metadata)}"
                )

            if "wordLabels" not in metadata:
                available_keys = list(metadata.keys())
                raise ValueError(
                    f"No 'wordLabels' key found in metadata file.\n"
                    f"Available keys: {available_keys}\n"
                    f"Please ensure the metadata file contains a 'wordLabels' array."
                )

            word_labels = metadata["wordLabels"]
            if not isinstance(word_labels, list):
                raise ValueError(
                    f"'wordLabels' must be a list, got {type(word_labels)}\n"
                    f"Please ensure 'wordLabels' is an array of strings."
                )

            if not word_labels:
                raise ValueError(
                    "'wordLabels' list cannot be empty.\n"
                    "Please provide at least one class label."
                )
            for i, label in enumerate(word_labels):
                if not isinstance(label, str):
                    raise ValueError(
                        f"All labels must be strings. Label at index {i} is {type(label)}: {label}"
                    )
                if not label.strip():
                    raise ValueError(
                        f"Label at index {i} is empty or contains only whitespace"
                    )

            self._labels = [label.strip() for label in word_labels]
            self._set_state(ConversionState.METADATA_LOADED)
            logging.info(f"✅ Loaded {len(self._labels)} labels: {self._labels}")

        except json.JSONDecodeError as e:
            self._set_state(ConversionState.FAILED)
            raise ValueError(
                f"Invalid JSON format in metadata file: {str(e)}\n"
                f"Please ensure the file contains valid JSON."
            ) from e
        except (ValueError, TypeError) as e:
            self._set_state(ConversionState.FAILED)
            raise ValueError(f"Metadata validation failed: {str(e)}") from e
        except Exception as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"Unexpected error while processing metadata: {str(e)}\n"
                f"Please check the file format and try again."
            ) from e

    def create_models(self) -> None:
        self._validate_state(ConversionState.METADATA_LOADED)

        logging.info("--- Step 3: Creating model architectures ---")

        # Validate that labels are loaded
        if not self._labels:
            self._set_state(ConversionState.FAILED)
            raise ValueError(
                "No class labels loaded. Cannot determine output layer size.\n"
                "Please ensure metadata loading completed successfully."
            )

        num_classes = len(self._labels)
        logging.info(f"🔄 Creating models for {num_classes} classes...")

        try:
            logging.info("🔄 Building original model architecture...")
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
                num_classes, activation="softmax", name="NewHeadDense"
            )(x_dense)
            self._original_model = tf.keras.Model(
                inputs=inputs, outputs=outputs, name="original_model"
            )

            if self._original_model is None:
                raise RuntimeError("Failed to create original model")

            logging.info(
                f"✅ Original model created with {self._original_model.count_params():,} parameters"
            )

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

            logging.info("🔄 Building optimized model architecture...")
            H, W = (x_before_flatten.shape[1], x_before_flatten.shape[2])
            if H is None or W is None:
                raise RuntimeError(
                    f"Cannot determine feature map dimensions. Got H={H}, W={W}\n"
                    f"This may indicate an issue with the model architecture."
                )

            x_conv1 = tf.keras.layers.Conv2D(
                2000,
                kernel_size=(H, W),
                activation="relu",
                name="dense_1_conv_replacement",
            )(x_before_flatten)
            x_flat1 = tf.keras.layers.Flatten(name="flatten_1")(x_conv1)
            x_dropout = tf.keras.layers.Dropout(0.25, name="dropout_1")(x_flat1)
            x_reshaped = tf.keras.layers.Reshape((1, 1, 2000))(x_dropout)
            x_conv2 = tf.keras.layers.Conv2D(
                num_classes,
                kernel_size=(1, 1),
                activation="softmax",
                name="NewHeadDense_conv_replacement",
            )(x_reshaped)
            outputs = tf.keras.layers.Flatten(name="output_flatten")(x_conv2)

            self._final_model = tf.keras.Model(
                inputs=inputs, outputs=outputs, name="optimized_model"
            )

            if self._original_model is None:
                raise RuntimeError("Failed to create original model architecture")
            if self._final_model is None:
                raise RuntimeError("Failed to create optimized model architecture")

            expected_output_shape = (None, num_classes)
            if self._original_model.output_shape != expected_output_shape:
                raise RuntimeError(
                    f"Original model output shape mismatch. Expected {expected_output_shape}, "
                    f"got {self._original_model.output_shape}"
                )
            if self._final_model.output_shape != expected_output_shape:
                raise RuntimeError(
                    f"Optimized model output shape mismatch. Expected {expected_output_shape}, "
                    f"got {self._final_model.output_shape}"
                )

            self._set_state(ConversionState.MODELS_CREATED)
            logging.info(
                f"✅ Original model created with {self._original_model.count_params():,} parameters"
            )
            logging.info(
                f"✅ Optimized model created with {self._final_model.count_params():,} parameters"
            )
            logging.info("✅ Model architectures created successfully!")

        except tf.errors.InvalidArgumentError as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"TensorFlow model creation failed due to invalid arguments: {str(e)}\n"
                f"This may indicate incompatible layer configurations."
            ) from e
        except Exception as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"Unexpected error during model creation: {str(e)}\n"
                f"Please check the model architecture configuration."
            ) from e

    def transfer_weights(self) -> None:
        self._validate_state(ConversionState.MODELS_CREATED)

        logging.info("--- Step 4: Loading and transferring weights ---")

        if not os.path.exists(self.keras_output_path):
            self._set_state(ConversionState.FAILED)
            raise FileNotFoundError(
                f"Keras model file not found: '{self.keras_output_path}'\n"
                f"Please ensure the TF.js to Keras conversion completed successfully."
            )

        if not os.access(self.keras_output_path, os.R_OK):
            self._set_state(ConversionState.FAILED)
            raise PermissionError(
                f"Keras model file is not readable: '{self.keras_output_path}'\n"
                f"Please check file permissions."
            )

        if os.path.getsize(self.keras_output_path) == 0:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"Keras model file is empty: '{self.keras_output_path}'\n"
                f"The file may be corrupted or the conversion may have failed."
            )

        try:
            logging.info(f"🔄 Loading weights from {self.keras_output_path}...")
            self._original_model.load_weights(self.keras_output_path)
            for layer_new in self._final_model.layers:
                if not layer_new.get_weights():
                    continue
                try:
                    source_layer = self._original_model.get_layer(name=layer_new.name)
                    layer_new.set_weights(source_layer.get_weights())
                except ValueError:
                    pass
            logging.info("✅ Weights loaded successfully")
            logging.info("🔄 Transferring weights to optimized model...")
            dense1_orig = self._original_model.get_layer("dense_1")
            conv1_new = self._final_model.get_layer("dense_1_conv_replacement")
            weights, biases = dense1_orig.get_weights()
            H, W, C_in, C_out = conv1_new.get_weights()[0].shape
            weights_reshaped = weights.reshape((H, W, C_in, C_out))
            conv1_new.set_weights([weights_reshaped, biases])
            logging.info(
                "  - Manually transferred: dense_1 -> dense_1_conv_replacement"
            )
            dense2_orig = self._original_model.get_layer("NewHeadDense")
            conv2_new = self._final_model.get_layer("NewHeadDense_conv_replacement")
            weights, biases = dense2_orig.get_weights()
            H, W, C_in, C_out = conv2_new.get_weights()[0].shape
            weights_reshaped = weights.reshape((H, W, C_in, C_out))
            conv2_new.set_weights([weights_reshaped, biases])
            logging.info(
                "  - Manually transferred: NewHeadDense -> NewHeadDense_conv_replacement"
            )

            self._set_state(ConversionState.WEIGHTS_TRANSFERRED)
            logging.info("✅ Weight transfer completed!")

        except tf.errors.InvalidArgumentError as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"TensorFlow weight loading failed: {str(e)}\n"
                f"The Keras model file may be corrupted or incompatible."
            ) from e
        except Exception as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(
                f"Unexpected error during weight transfer: {str(e)}\n"
                f"Please check the model files and try again."
            ) from e

    def quantize_and_convert(self) -> None:
        self._validate_state(ConversionState.WEIGHTS_TRANSFERRED)

        logging.info("--- Step 5: Quantizing and converting to TFLite ---")

        try:

            @tf.function
            def model_func(input_tensor):
                return self._final_model(input_tensor)

            fixed_input_spec = tf.TensorSpec(shape=[1, 43, 232, 1], dtype=tf.float32)
            concrete_func = model_func.get_concrete_function(fixed_input_spec)

            converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            self._tflite_model = converter.convert()

            os.makedirs(os.path.dirname(self.tflite_output_path), exist_ok=True)

            with open(self.tflite_output_path, "wb") as f:
                f.write(self._tflite_model)

            logging.info(f"✅ Quantized TFLite model saved: {self.tflite_output_path}")

            interpreter = tf.lite.Interpreter(model_content=self._tflite_model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()[0]
            output_details = interpreter.get_output_details()[0]

            self._model_info = {
                "input_shape": input_details["shape"].tolist(),
                "input_dtype": str(input_details["dtype"]),
                "input_scale": float(input_details["quantization"][0]),
                "input_zero_point": int(input_details["quantization"][1]),
                "output_shape": output_details["shape"].tolist(),
                "output_dtype": str(output_details["dtype"]),
                "output_scale": float(output_details["quantization"][0]),
                "output_zero_point": int(output_details["quantization"][1]),
                "model_size": len(self._tflite_model),
                "num_classes": len(self._labels),
                "class_labels": self._labels,
            }

            self._set_state(ConversionState.QUANTIZED)
            logging.info(
                f"✅ Model info extracted: {self._model_info['input_shape']} -> {self._model_info['output_shape']}"
            )

        except Exception as e:
            self._set_state(ConversionState.FAILED)
            raise RuntimeError(f"Quantization and conversion failed: {str(e)}") from e

    def get_model_info(self) -> Dict[str, Any]:
        if self._conversion_state != ConversionState.QUANTIZED:
            raise RuntimeError(
                "Model conversion must be completed before accessing model info"
            )
        return self._model_info.copy()

    def get_labels(self) -> List[str]:
        if self._conversion_state.value in [
            ConversionState.INITIALIZED.value,
            ConversionState.FAILED.value,
        ]:
            raise RuntimeError("Metadata must be loaded before accessing labels")
        return self._labels.copy()

    def get_tflite_model(self) -> bytes:
        if self._conversion_state != ConversionState.QUANTIZED:
            raise RuntimeError(
                "Model conversion must be completed before accessing TFLite model"
            )
        return self._tflite_model

    def run(self) -> None:
        logging.info("🚀 Starting Optimized AcousticsLab Model Conversion Pipeline")
        logging.info("=" * 65)

        try:
            self.convert_tfjs_to_keras()
            self.load_metadata()
            self.create_models()
            self.transfer_weights()
            self.quantize_and_convert()

            self._set_state(ConversionState.COMPLETED)

            logging.info("\n" + "=" * 65)
            logging.info("🎉 Conversion completed successfully!")
            logging.info("📁 Generated files:")
            logging.info(f"   - TFLite model: {self.tflite_output_path}")
            logging.info("\n📋 Model summary:")
            logging.info(f"   - Input shape: {self._model_info['input_shape']}")
            logging.info(f"   - Output classes: {self._model_info['num_classes']}")
            logging.info(f"   - Model size: {self._model_info['model_size']:,} bytes")
            logging.info(f"   - Labels: {', '.join(self._labels)}")
            logging.info("\n✨ TFLite model ready for deployment!")

        except Exception as e:
            self._set_state(ConversionState.FAILED)
            logging.info(f"\n❌ Conversion failed: {str(e)}")
            raise


def _validate_path_exists(path: str, path_type: str) -> None:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(
            f"{path_type} not found: '{path}'\n"
            f"Please ensure the path exists and is accessible."
        )

    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"{path_type} is not readable: '{path}'\nPlease check file permissions."
        )


def _validate_output_path(path: str, path_type: str) -> None:
    path_obj = Path(path)
    output_dir = path_obj.parent

    # Create directory if it doesn't exist
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(
            f"Cannot create output directory for {path_type}: '{output_dir}'\n"
            f"Please check directory permissions."
        )

    # Check if directory is writable
    if not os.access(output_dir, os.W_OK):
        raise PermissionError(
            f"Output directory for {path_type} is not writable: '{output_dir}'\n"
            f"Please check directory permissions."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Model Converter for AcousticsLab. Converts TF.js models to TFLite."
    )

    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Simplified usage: specify TF.js model and metadata files in order. "
        "Example: --files model.json metadata.json",
    )

    parser.add_argument(
        "--tfjs_model",
        type=str,
        default=str(DEFAULT_TFJS_MODEL_PATH),
        help=f"Path to the input TF.js model.json file. Default: {DEFAULT_TFJS_MODEL_PATH}",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(DEFAULT_METADATA_PATH),
        help=f"Path to the input metadata.json file. Default: {DEFAULT_METADATA_PATH}",
    )
    parser.add_argument(
        "--keras_output",
        type=str,
        default=str(DEFAULT_KERAS_OUTPUT_PATH),
        help=f"Path to save the intermediate Keras .h5 model. Default: {DEFAULT_KERAS_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--tflite_output",
        type=str,
        default=str(DEFAULT_TFLITE_OUTPUT_PATH),
        help=f"Path to save the final quantized .tflite model. Default: {DEFAULT_TFLITE_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=str(DEFAULT_WAV_INPUT_DIR),
        help=f"Directory containing input .wav files for quantization calibration. Default: {DEFAULT_WAV_INPUT_DIR}",
    )
    parser.add_argument(
        "--preprocessing_model",
        type=str,
        default=str(DEFAULT_PREPROCESSING_MODEL_PATH),
        help=f"Path to the TFLite model used for audio preprocessing (feature extraction). Default: {DEFAULT_PREPROCESSING_MODEL_PATH}",
    )

    args = parser.parse_args()

    if args.files:
        if len(args.files) < 2:
            parser.error(
                "--files requires at least 2 arguments: TF.js model and metadata file"
            )
        elif len(args.files) > 2:
            parser.error(
                "--files accepts exactly 2 arguments: TF.js model and metadata file"
            )

        args.tfjs_model = args.files[0]
        args.metadata = args.files[1]
        tfjs_path = Path(args.tfjs_model)
        base_name = tfjs_path.stem
        output_dir = tfjs_path.parent

        args.keras_output = str(output_dir / f"{base_name}_converted.h5")
        args.tflite_output = str(output_dir / f"{base_name}_quantized.tflite")

    try:
        _validate_path_exists(args.tfjs_model, "TF.js model file")
        _validate_path_exists(args.metadata, "Metadata file")
        _validate_path_exists(args.wav_dir, "WAV input directory")
        _validate_path_exists(args.preprocessing_model, "Preprocessing model file")
        _validate_output_path(args.keras_output, "Keras output file")
        _validate_output_path(args.tflite_output, "TFLite output file")

        logging.info("✅ All input paths validated successfully")
        logging.info("📁 Input files:")
        logging.info(f"   - TF.js model: {args.tfjs_model}")
        logging.info(f"   - Metadata: {args.metadata}")
        logging.info(f"   - WAV directory: {args.wav_dir}")
        logging.info(f"   - Preprocessing model: {args.preprocessing_model}")
        logging.info("📁 Output files:")
        logging.info(f"   - Keras model: {args.keras_output}")
        logging.info(f"   - TFLite model: {args.tflite_output}")

    except (FileNotFoundError, PermissionError) as e:
        logging.error(f"❌ Path validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Unexpected error during path validation: {e}")
        sys.exit(1)

    try:
        converter = ModelConverter(
            tfjs_model_path=args.tfjs_model,
            metadata_path=args.metadata,
            keras_output_path=args.keras_output,
            tflite_output_path=args.tflite_output,
            wav_input_dir=args.wav_dir,
            preprocessing_model_path=args.preprocessing_model,
        )
        converter.run()
    except KeyboardInterrupt:
        logging.info("\n⚠️  Conversion interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logging.critical(f"❌ An unrecoverable error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
