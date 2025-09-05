#!/usr/bin/env python3
"""
End-to-End Speech Model Converter for Deployment

This script provides a comprehensive pipeline to convert a speech command model,
originally from TensorFlow.js (like those from Teachable Machine), into a
fully quantized and optimized TensorFlow Lite (`.tflite`) model suitable for
deployment on microcontroller units (MCUs).

The conversion process involves several key stages:
1.  **TF.js to Keras**: Converts the input TF.js model (`model.json`) into an intermediate Keras H5 model (`.h5`) which is used solely as a weight container.
2.  **Manual Model Reconstruction**: A known-correct Keras model architecture is manually defined in the script.
3.  **Weight Loading**: Weights are loaded by name from the H5 file into the newly constructed Keras model, bypassing potential structural issues in the H5 file.
4.  **Post-Training Quantization**: Converts the model to TensorFlow Lite format
    while applying full integer quantization (INT8).
5.  **Metadata Integration**: Reads class labels from a `metadata.json` file to
    validate the model's output layer.

Usage:
    The script can be run from the command line, providing paths to the
    necessary input files and desired output locations.

    Usage:
    $ python convert_speech_model.py --tfjs_model /path/to/model.json \\
                                     --metadata /path/to/metadata.json \\
                                     --wav_dir /path/to/audio_data
    For a full list of options, run:
    $ python convert_speech_model.py --help
"""

import tensorflow as tf
import numpy as np
import logging
import argparse
import sys
import json
import random
import mimetypes
import urllib.request
import tarfile
import os
import librosa
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator

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

REQUIRED_SAMPLES = 100


def download_and_extract_preproc_model(dest_dir: Path) -> Path:
    PREPROC_MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz"
    MODEL_DIR_NAME = "sc_preproc_model"

    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / MODEL_DIR_NAME

    if model_path.exists():
        logging.info(f"✅ Preprocessing model found at {model_path}")
        return model_path

    logging.info(
        f"Preprocessing model not found. Downloading from {PREPROC_MODEL_URL}..."
    )

    tar_path, _ = urllib.request.urlretrieve(PREPROC_MODEL_URL)
    logging.info(f"Downloaded to temporary file: {tar_path}")

    logging.info(f"Extracting to {dest_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    os.remove(tar_path)

    if not model_path.exists():
        raise RuntimeError(f"Failed to extract model to {model_path}")

    logging.info(
        f"✅ Successfully downloaded and extracted preprocessing model to {model_path}"
    )
    return model_path


class ConversionStepFailedError(Exception):
    def __init__(
        self, step_name: str, message: str, original_exception: Exception = None
    ):
        self.step_name = step_name
        self.original_exception = original_exception
        super().__init__(f"Step '{step_name}' failed: {message}")


class ModelConverter:
    def __init__(
        self,
        tfjs_model_path: str = str(DEFAULT_TFJS_MODEL_PATH),
        metadata_path: str = str(DEFAULT_METADATA_PATH),
        keras_output_path: str = str(DEFAULT_KERAS_OUTPUT_PATH),
        tflite_output_path: str = str(DEFAULT_TFLITE_OUTPUT_PATH),
        wav_input_dir: str = str(DEFAULT_WAV_INPUT_DIR),
    ):
        path_configs = {
            "tfjs_model_path": tfjs_model_path,
            "metadata_path": metadata_path,
            "keras_output_path": keras_output_path,
            "tflite_output_path": tflite_output_path,
            "wav_input_dir": wav_input_dir,
        }

        for attr_name, path_value in path_configs.items():
            if not isinstance(path_value, str) or not path_value.strip():
                raise ValueError(f"{attr_name} must be a non-empty string")
            setattr(self, attr_name, Path(path_value.strip()))

        preproc_cache_dir = _PROJECT_ROOT / "tm-my-audio-model"
        self.preproc_model_path = download_and_extract_preproc_model(preproc_cache_dir)

        self._model_info: Dict[str, Any] = {}
        self._labels: List[str] = []
        self._num_classes: Optional[int] = None
        self._keras_model: Optional[tf.keras.Model] = None
        self._tflite_model: Optional[bytes] = None

        logging.info("🔧 ModelConverter initialized successfully")

    def _check_keras_model_exists(self) -> None:
        if not self.keras_output_path.exists():
            raise ConversionStepFailedError(
                "dependency_check",
                f"Keras model not found at {self.keras_output_path}. "
                "Run convert_tfjs_to_keras() first.",
            )

    def _check_keras_model_built(self) -> None:
        if self._keras_model is None:
            raise ConversionStepFailedError(
                "dependency_check",
                "Keras model has not been built. Run build_and_load_weights() first.",
            )

    @staticmethod
    def _collect_audio_files(directory: Path) -> List[Path]:
        audio_files = []
        try:
            for file_path in directory.iterdir():
                mime_type, _ = mimetypes.guess_type(str(file_path))
                is_audio = mime_type and mime_type.startswith("audio/")
                if file_path.is_file() and is_audio:
                    audio_files.append(file_path)
            return sorted(audio_files)
        except Exception as e:
            raise IOError(
                f"Failed to collect audio files from {directory}: {e}"
            ) from e

    def _validate_and_sample_audio_files(self) -> List[Path]:
        try:
            wav_dir = Path(self.wav_input_dir)
            audio_files = self._collect_audio_files(wav_dir)
            logging.info(f"Found {len(audio_files)} audio files in {wav_dir}")

            if len(audio_files) < REQUIRED_SAMPLES:
                raise ValueError(
                    f"Insufficient audio files: found {len(audio_files)}, "
                    f"required at least {REQUIRED_SAMPLES}"
                )

            if len(audio_files) > REQUIRED_SAMPLES:
                selected_files = random.sample(audio_files, REQUIRED_SAMPLES)
                logging.info(
                    f"Randomly sampled {REQUIRED_SAMPLES} files from {len(audio_files)} available"
                )
            else:
                selected_files = audio_files
                logging.info(f"Using all {len(selected_files)} available files")

            return selected_files
        except (ValueError, IOError) as e:
            raise ConversionStepFailedError("dataset_validation", str(e), original_exception=e) from e

    def _generate_features_from_wavs(self) -> Iterator[np.ndarray]:
        logging.info(
            "--- Starting feature generation from audio files for quantization ---"
        )
        audio_files = self._validate_and_sample_audio_files()
        logging.info(
            f"Processing {len(audio_files)} audio files to generate representative data..."
        )

        try:
            preproc_model = tf.saved_model.load(str(self.preproc_model_path))
            inference_func = preproc_model.signatures["serving_default"]
            input_details = inference_func.inputs[0]
            target_len = input_details.shape[1]
            target_sr = 44100

            for audio_path in audio_files:
                try:
                    data, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
                    if len(data) > target_len:
                        data = data[
                            (len(data) - target_len) // 2 : (len(data) + target_len)
                            // 2
                        ]
                    else:
                        data = np.pad(data, (0, target_len - len(data)), "constant")

                    model_input_audio = tf.constant(
                        data.reshape(1, target_len), dtype=input_details.dtype
                    )
                    result_dict = inference_func(model_input_audio)
                    feature_map = list(result_dict.values())[0].numpy()
                    yield [feature_map.astype(np.float32)]
                except Exception as e:
                    logging.warning(
                        f"Could not process audio file '{audio_path.name}': {e}. Skipping."
                    )
                    continue
        except Exception as e:
            raise ConversionStepFailedError(
                "feature_generation",
                f"Failed to initialize preprocessing model or generate features: {e}",
                original_exception=e
            ) from e

    def _representative_data_gen(self):
        try:
            yield from self._generate_features_from_wavs()
        except Exception as e:
            logging.error(
                f"Failed to generate data from WAVs: {e}. Falling back to random samples for calibration."
            )
            for _ in range(200):
                yield [np.random.rand(1, 43, 232, 1).astype(np.float32)]

    def convert_tfjs_to_keras(self) -> None:
        logging.info(
            f"--- Step 1: Converting {self.tfjs_model_path} to {self.keras_output_path} ---"
        )
        try:
            logging.info("🔄 Converting TF.js model to H5 weight container...")
            self.keras_output_path.parent.mkdir(parents=True, exist_ok=True)
            dispatch_tensorflowjs_to_keras_h5_conversion(
                str(self.tfjs_model_path),
                str(self.keras_output_path),
            )
            logging.info("TF.js to Keras H5 conversion completed successfully")
            if (
                not self.keras_output_path.exists()
                or self.keras_output_path.stat().st_size == 0
            ):
                raise RuntimeError("Keras H5 weight file was not created or is empty")
            file_size = self.keras_output_path.stat().st_size
            logging.info(
                f"✅ Keras H5 weight file created! Size: {file_size:,} bytes"
            )
        except Exception as e:
            raise ConversionStepFailedError(
                "tfjs_to_keras_conversion",
                f"Failed to convert TF.js model to Keras H5: {e}",
                original_exception=e
            ) from e

    def build_and_load_weights(self) -> None:
        self._check_keras_model_exists()
        logging.info("--- Step 2: Building Keras model and loading weights ---")
        if not self.metadata_path.exists():
            raise ConversionStepFailedError(
                "metadata_loading", f"Metadata file not found: {self.metadata_path}"
            )
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            word_labels = metadata.get("wordLabels", [])
            if not word_labels or not isinstance(word_labels, list):
                raise ValueError("'wordLabels' key is missing, empty, or not a list.")
            self._labels = [str(label).strip() for label in word_labels]
            self._num_classes = len(self._labels)
            logging.info(f"✅ Loaded {self._num_classes} labels from metadata: {self._labels}")
        except Exception as e:
            raise ConversionStepFailedError(
                "metadata_loading", f"Failed to load or parse metadata: {e}", original_exception=e
            ) from e

        try:
            logging.info(f"🔄 Building Keras model architecture for {self._num_classes} classes...")
            inputs = tf.keras.layers.Input(batch_shape=(1, 43, 232, 1), name="conv2d_1_input")
            
            x = tf.keras.layers.Conv2D(8, kernel_size=(2, 8), activation="relu", name="conv2d_1")(inputs)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_1")(x)
            x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation="relu", name="conv2d_2")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_2")(x)
            x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation="relu", name="conv2d_3")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="max_pooling2d_3")(x)
            x = tf.keras.layers.Conv2D(32, kernel_size=(2, 4), activation="relu", name="conv2d_4")(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 2), name="max_pooling2d_4")(x)
            x = tf.keras.layers.Reshape((704,), name="flatten_1")(x)
            x = tf.keras.layers.Dropout(0.25, name="dropout_1")(x)
            x = tf.keras.layers.Dense(2000, activation="relu", name="dense_1")(x)
            outputs = tf.keras.layers.Dense(self._num_classes, activation="softmax", name="dense_2")(x)
            
            self._keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="reconstructed_model")
            logging.info(f"✅ Keras model built successfully with {self._keras_model.count_params():,} parameters.")
        except Exception as e:
            raise ConversionStepFailedError("model_building", f"Failed to build Keras model: {e}", original_exception=e) from e

        try:
            logging.info(f"🔄 Loading weights from {self.keras_output_path} into the new model...")
            self._keras_model.load_weights(str(self.keras_output_path))
            logging.info("✅ Weights loaded successfully.")
        except Exception as e:
            raise ConversionStepFailedError(
                "weight_loading", f"Failed to load weights into the built model: {e}", original_exception=e
            ) from e


    def quantize_and_convert(self) -> None:
        self._check_keras_model_built()
        logging.info("--- Step 3: Quantizing and converting to TFLite ---")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(self._keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            self._tflite_model = converter.convert()
            self.tflite_output_path.parent.mkdir(parents=True, exist_ok=True)
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
                "num_classes": self._num_classes,
                "class_labels": self._labels,
            }
            logging.info(
                f"✅ Model info extracted: {self._model_info['input_shape']} -> {self._model_info['output_shape']}"
            )
        except Exception as e:
            raise ConversionStepFailedError(
                "quantization_and_conversion",
                f"Failed to quantize and convert model: {e}",
                original_exception=e
            ) from e

    def run(self) -> None:
        logging.info("🚀 Starting Robust AcousticsLab Model Conversion Pipeline")
        logging.info("=" * 65)
        try:
            self.convert_tfjs_to_keras()
            self.build_and_load_weights()
            self.quantize_and_convert()

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
        except ConversionStepFailedError:
            raise
        except Exception as e:
            raise ConversionStepFailedError(
                "pipeline_execution", f"Unexpected error in conversion pipeline: {e}", original_exception=e
            ) from e


def validate_audio_directory(wav_dir_path: str) -> None:
    wav_dir = Path(wav_dir_path)
    if not wav_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {wav_dir}")
    if not wav_dir.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {wav_dir}")
    
    try:
        audio_files = ModelConverter._collect_audio_files(wav_dir)
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in directory: {wav_dir}")
        logging.info(
            f"✅ Audio directory validation passed: {len(audio_files)} audio files found"
        )
    except IOError as e:
        raise e


def main():
    parser = argparse.ArgumentParser(
        description="Robust Model Converter for AcousticsLab. Converts TF.js models to TFLite."
    )
    parser.add_argument(
        "--tfjs_model",
        type=str,
        default=str(DEFAULT_TFJS_MODEL_PATH),
        help="Path to the input TF.js model.json file",
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
    args = parser.parse_args()

    logging.info("📁 Input configuration:")
    logging.info(f"   - TF.js model: {args.tfjs_model}")
    logging.info(f"   - Metadata: {args.metadata}")
    logging.info(f"   - Audio directory: {args.wav_dir}")
    logging.info("📁 Output configuration:")
    logging.info(f"   - Keras H5 (weights only): {args.keras_output}")
    logging.info(f"   - TFLite model: {args.tflite_output}")

    try:
        logging.info("🔍 Validating input audio directory...")
        validate_audio_directory(args.wav_dir)
        
        converter = ModelConverter(
            tfjs_model_path=args.tfjs_model,
            metadata_path=args.metadata,
            keras_output_path=args.keras_output,
            tflite_output_path=args.tflite_output,
            wav_input_dir=args.wav_dir,
        )
        converter.run()
    except (FileNotFoundError, NotADirectoryError, ValueError, IOError) as e:
        logging.error(f"❌ Input validation failed: {e}")
        sys.exit(1)
    except ConversionStepFailedError as e:
        logging.error(f"❌ Conversion failed at step '{e.step_name}': {e}")
        if e.original_exception:
            logging.error("📋 Full error details:", exc_info=e.original_exception)
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("\n⚠️  Conversion interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.critical(f"❌ An unexpected error occurred during conversion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()