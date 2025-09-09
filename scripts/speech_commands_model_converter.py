import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional
import logging

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

CWD = Path.cwd()
DEFAULT_TFJS_MODEL = CWD / "tm-my-audio-model"
DEFAULT_OUT_MODEL = CWD / "tm-my-audio-model.tflite"


def _resolve_tfjs_input_path(tfjs_path: Path) -> Path:
    if tfjs_path.is_dir():
        cand = tfjs_path / "model.json"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"model.json not found in {tfjs_path}")
    if tfjs_path.name == "model.json" and tfjs_path.exists():
        return tfjs_path
    raise FileNotFoundError(
        f"Expected a TFJS layers model directory (with model.json) or a model.json file, got: {tfjs_path}"
    )


def _run_tfjs_to_saved_model(tfjs_model_path_or_dir: Path, saved_model_out_dir: Path):
    model_json = _resolve_tfjs_input_path(tfjs_model_path_or_dir)
    saved_model_out_dir.parent.mkdir(parents=True, exist_ok=True)
    dispatch_tensorflowjs_to_keras_saved_model_conversion(
        str(model_json),
        str(saved_model_out_dir),
    )
    if not saved_model_out_dir.exists():
        print(f"ERROR: SavedModel not found at {saved_model_out_dir}", file=sys.stderr)
        raise FileNotFoundError(saved_model_out_dir)


def _make_representative_dataset_fn(
    npy_path: Path, input_shape: tuple[int, ...], samples: int = 100
) -> Callable[[], Iterable[list]]:
    if npy_path is None:

        def syn_ds():
            for _ in range(samples):
                yield [np.random.random(input_shape).astype("float32")]

        return syn_ds

    arr = np.load(str(npy_path))
    if arr.ndim == len(input_shape):
        pass
    elif arr.ndim + 1 == len(input_shape):
        arr = arr.reshape((-1,) + tuple(input_shape[1:]))
    else:
        raise ValueError(
            f"Representative data shape {arr.shape} does not match model input shape {input_shape}"
        )

    def rep_ds():
        max_samples = min(samples, arr.shape[0])
        for i in range(max_samples):
            sample = arr[i : i + 1].astype("float32")
            yield [sample]

    return rep_ds


def _get_concrete_input_shape(model: tf.Module) -> Optional[tuple[int, ...]]:
    concrete = None
    signatures = model.signatures
    if "serving_default" in signatures:
        concrete = signatures["serving_default"]
    elif len(signatures) > 0:
        concrete = next(iter(signatures.values()))
    if concrete is not None and concrete.inputs:
        tshape = concrete.inputs[0].shape.as_list()
        if tshape and tshape[0] is None:
            tshape[0] = 1
        return tuple(int(x) for x in tshape)
    return None


def convert(
    tfjs_dir: Path,
    output_tflite: Path,
    quantize: str = "none",
    rep_data: Optional[Path] = None,
    input_shape_override: Optional[str] = None,
) -> Path:
    tfjs_dir = tfjs_dir.resolve()
    if not (tfjs_dir / "model.json").exists():
        raise FileNotFoundError(f"model.json not found in {tfjs_dir}")
    output_tflite = output_tflite.resolve()
    tmpdir = Path(tempfile.mkdtemp(prefix="tfjs2tflite_"))
    saved_model_dir = tmpdir / "saved_model"

    converter = None
    model = None
    input_shape_inferred = None
    quantize = quantize.lower().strip()

    try:
        _run_tfjs_to_saved_model(tfjs_dir, saved_model_dir)
        model = tf.saved_model.load(str(saved_model_dir))

        match quantize:
            case "dynamic":
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    str(saved_model_dir)
                )

            case "float16":
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    str(saved_model_dir)
                )
                converter.target_spec.supported_types = [tf.float16]

            case "int8":
                if input_shape_inferred is None:
                    input_shape = _get_concrete_input_shape(model)
                else:
                    if not input_shape_override:
                        raise ValueError("Cannot infer input shape.")
                    dims = tuple(
                        int(x.strip())
                        for x in input_shape_override.split(",")
                        if x.strip()
                    )
                    if len(dims) == 0:
                        raise ValueError("--input-shape provided but empty")
                    if dims[0] != 1:
                        dims = (1,) + dims
                    input_shape = dims

                concrete_func = model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                concrete_func.inputs[0].set_shape(input_shape)
                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [concrete_func]
                )
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                rep_fn = _make_representative_dataset_fn(rep_data, input_shape)
                converter.representative_dataset = rep_fn

            case _:
                raise ValueError(f"Unsupported quantization mode: {quantize}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to convert TFJS model to TFLite model due to: {e}"
        ) from e

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_tflite.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tflite, "wb") as f:
        f.write(tflite_model)

    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    return output_tflite


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Speech Commands TFJS model to TFLite"
    )
    p.add_argument(
        "--tfjs-dir",
        type=Path,
        default=DEFAULT_TFJS_MODEL,
        help="Directory with model.json and weights.bin",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT_MODEL,
        help="Output .tflite file path",
    )
    p.add_argument(
        "--quantize",
        choices=["dynamic", "float16", "int8"],
        default="dynamic",
        help="Quantization mode (default: dynamic)",
    )
    p.add_argument(
        "--rep-data",
        type=Path,
        default=None,
        help="Path to .npy representative dataset for int8 quantization",
    )
    p.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Comma-separated input shape excluding batch, e.g. '43,232,1'",
    )

    args = p.parse_args(argv)

    out = convert(
        args.tfjs_dir, args.output, args.quantize, args.rep_data, args.input_shape
    )
    size_kb = os.path.getsize(out) / 1024.0
    logging.info(f"Converted successfully, output: {out} ({size_kb:.1f} KB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
