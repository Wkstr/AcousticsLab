import argparse
import logging
import mimetypes
import os
import random
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Optional, List

import librosa
import numpy as np
import tensorflow as tf
from tensorflowjs.converters.converter import (
    dispatch_tensorflowjs_to_keras_saved_model_conversion,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
DEFAULT_TFJS_MODEL_DIR = _PROJECT_ROOT / "tm-my-audio-model"
DEFAULT_WAV_INPUT_DIR = _PROJECT_ROOT / "datasound"
DEFAULT_TFLITE_OUTPUT_PATH = _PROJECT_ROOT / "tm-my-audio-model.tflite"
REQUIRED_SAMPLES = 100


def _download_and_extract_preproc_model(dest_dir: Path) -> Path:
    PREPROC_MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz"
    MODEL_DIR_NAME = "sc_preproc_model"

    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / MODEL_DIR_NAME

    if model_path.exists():
        logging.info(f"Preprocessing model found at {model_path}")
        return model_path

    logging.info(
        f"Preprocessing model not found. Downloading from {PREPROC_MODEL_URL}..."
    )
    try:
        tar_path, _ = urllib.request.urlretrieve(PREPROC_MODEL_URL)
        logging.info(f"Downloaded to temporary file: {tar_path}")

        logging.info(f"Extracting to {dest_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=dest_dir)
        os.remove(tar_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download or extract preprocessing model: {e}") from e

    if not model_path.exists():
        raise RuntimeError(f"Failed to extract model to {model_path}")

    logging.info(
        f"Successfully downloaded and extracted preprocessing model to {model_path}"
    )
    return model_path


def _collect_audio_files(directory: Path) -> List[Path]:
    audio_files = []
    logging.info(f"Scanning for audio files in {directory}...")
    for file_path in directory.iterdir():
        mime_type, _ = mimetypes.guess_type(str(file_path))
        is_audio = mime_type and mime_type.startswith("audio/")
        if file_path.is_file() and is_audio:
            audio_files.append(file_path)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in directory: {directory}")
    return sorted(audio_files)


def _make_audio_dataset_generator(
    wav_dir: Path, preproc_model_path: Path) -> Callable[[], Iterable[list]]:
    def generator() -> Iterable[list]:
        logging.info("--- Starting feature generation from audio for quantization ---")
        try:
            audio_files = _collect_audio_files(wav_dir)
            logging.info(f"Found {len(audio_files)} audio files.")

            if len(audio_files) < REQUIRED_SAMPLES:
                logging.warning(
                    f"Found {len(audio_files)} audio files, but at least {REQUIRED_SAMPLES} are recommended for robust quantization."
                )
                selected_files = audio_files
            else:
                selected_files = random.sample(audio_files, REQUIRED_SAMPLES)
                logging.info(f"Randomly sampled {len(selected_files)} files for calibration.")

            preproc_model = tf.saved_model.load(str(preproc_model_path))
            inference_func = preproc_model.signatures["serving_default"]
            input_details = inference_func.inputs[0]
            target_len = input_details.shape[1]
            target_sr = 44100

            for audio_path in selected_files:
                try:
                    data, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
                    if len(data) > target_len:
                        start = (len(data) - target_len) // 2
                        data = data[start : start + target_len]
                    else:
                        data = np.pad(data, (0, target_len - len(data)), "constant")

                    model_input = tf.constant(data.reshape(1, target_len), dtype=input_details.dtype)
                    result = inference_func(model_input)
                    feature_map = list(result.values())[0].numpy()
                    yield [feature_map.astype(np.float32)]
                except Exception as e:
                    logging.warning(f"Could not process '{audio_path.name}': {e}. Skipping.")
                    continue
        except Exception as e:
            logging.error(f"Fatal error during feature generation: {e}")
            logging.error("Falling back to synthetic random data for calibration.")
            for _ in range(REQUIRED_SAMPLES):
                yield [np.random.rand(1, 43, 232, 1).astype(np.float32)]

    return generator


def convert(
    tfjs_dir: Path,
    output_tflite: Path,
    quantize: str = "dynamic",
    wav_dir: Optional[Path] = None,
) -> Path:
    tfjs_model_path = tfjs_dir / "model.json"
    if not tfjs_model_path.exists():
        raise FileNotFoundError(f"model.json not found in {tfjs_dir}")

    output_tflite = output_tflite.resolve()
    quantize = quantize.lower().strip()

    with tempfile.TemporaryDirectory(prefix="tfjs2tflite_") as tmpdir:
        saved_model_dir = Path(tmpdir) / "saved_model"

        logging.info(f"--- Step 1: Converting {tfjs_model_path} to SavedModel format ---")
        try:
            dispatch_tensorflowjs_to_keras_saved_model_conversion(
                str(tfjs_model_path), str(saved_model_dir)
            )
            logging.info(f"SavedModel created at {saved_model_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to convert TF.js to SavedModel: {e}") from e

        logging.info(f"--- Step 2: Converting to TFLite with '{quantize}' quantization ---")
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantize == "float16":
            converter.target_spec.supported_types = [tf.float16]
            logging.info("Enabled float16 quantization.")
        
        elif quantize == "int8":
            if not wav_dir or not wav_dir.is_dir():
                raise ValueError(f"For 'int8' quantization, a valid --wav-dir is required.")
            
            preproc_cache_dir = Path(tmpdir) / "preproc_model_cache"
            preproc_model_path = _download_and_extract_preproc_model(preproc_cache_dir)
            
            converter.representative_dataset = _make_audio_dataset_generator(
                wav_dir, preproc_model_path
            )
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            logging.info("Enabled full integer (int8) quantization using audio dataset.")

        elif quantize == "dynamic":
            logging.info("Enabled dynamic range quantization.")

        else:
            raise ValueError(f"Unsupported quantization mode: {quantize}")

        try:
            tflite_model = converter.convert()
            logging.info("TFLite conversion successful.")
        except Exception as e:
            raise RuntimeError(f"TFLite conversion failed: {e}") from e

    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    return output_tflite


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Speech Commands TFJS model to TFLite"
    )
    p.add_argument(
        "--tfjs-dir",
        type=Path,
        default=DEFAULT_TFJS_MODEL_DIR,
        help=f"Directory with model.json, metadata.json, and weights.bin. Default: {DEFAULT_TFJS_MODEL_DIR}",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TFLITE_OUTPUT_PATH,
        help=f"Output .tflite file path. Default: {DEFAULT_TFLITE_OUTPUT_PATH}",
    )
    p.add_argument(
        "--quantize",
        choices=["dynamic", "float16", "int8"],
        default="int8",
        help="Quantization mode. 'int8' is recommended for MCUs. Default: int8",
    )
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=DEFAULT_WAV_INPUT_DIR,
        help=f"Directory of .wav files for 'int8' quantization calibration. Default: {DEFAULT_WAV_INPUT_DIR}",
    )

    args = p.parse_args(argv)

    try:
        logging.info("Starting Model Conversion Pipeline")
        logging.info(f"   - TF.js Source: {args.tfjs_dir}")
        logging.info(f"   - TFLite Output: {args.output}")
        logging.info(f"   - Quantization: {args.quantize}")
        if args.quantize == 'int8':
            logging.info(f"   - Audio Data for Calibration: {args.wav_dir}")
        
        out_path = convert(
            args.tfjs_dir, args.output, args.quantize, args.wav_dir
        )
        size_kb = os.path.getsize(out_path) / 1024.0
        logging.info("=" * 50)
        logging.info(f"Conversion successful!")
        logging.info(f"   - Output model: {out_path} ({size_kb:.1f} KB)")
        logging.info("=" * 50)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logging.error(f"ERROR: {e}")
        return 1
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())