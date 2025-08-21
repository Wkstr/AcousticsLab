# Audio Sampler TS

A tiny TypeScript CLI that reads audio frames from a serial device, decodes ADPCM via WASM module from `adpcm_rs`, and writes PCM audio to a WAV file.

- Serial default: `/dev/ttyACM0`
- Baud default: `921600`
- Output: `output.wav` (16‑bit PCM, mono by default)
- Decoder: `tools/adpcm_rs/pkg` (WASM, Node target)

## Packet format

The tool expects line-delimited frames in this format from the serial port:

```
SP: <predictor> <step_ndex> <base64_adpcm>
```

- `predictor`: initial PCM predictor for the ADPCM frame (int, typically fits in int16)
- `step_ndex`: IMA ADPCM step index (0..88)
- `base64_adpcm`: base64-encoded ADPCM bytes for the frame

Each line is decoded to 16‑bit PCM using the WASM decoder and appended to the WAV stream.

## Quick start

```sh
cd tools/audio_sampler_ts
npm install
npm run build

# Usage
npm run dev -- --help

# List ports
npm run list

# Decode and write to WAV (default options)
npm start -- --port /dev/ttyACM0 --baud 921600 --output output.wav

# Custom audio params
npm start -- --port /dev/ttyACM0 -sr 44100 -ch 1 -o capture.wav

# Dev (no build, uses ts-node)
npm run dev -- --port /dev/ttyACM0 --baud 921600
```

WAV is finalized on exit (Ctrl+C). On SIGINT/SIGTERM, the writer is closed and the file is saved.
