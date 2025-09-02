import argparse
import base64
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import serial
except ImportError:
    print(
        "Missing dependency pyserial. Please pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise

try:
    from opuslib import Decoder
except ImportError:
    print(
        "Missing dependency opuslib. Please pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Record base64 Opus packets from serial and decode to WAV"
    )
    p.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port path (default: /dev/ttyACM0)",
    )
    p.add_argument(
        "--baud", type=int, default=921600, help="Serial baud rate (default: 921600)"
    )
    p.add_argument("--outfile", default="output.wav", help="Output WAV file path")
    p.add_argument(
        "--channels", type=int, default=1, help="Number of audio channels (default: 1)"
    )
    p.add_argument(
        "--rate", type=int, default=48000, help="Sample rate Hz (default: 48000)"
    )
    p.add_argument(
        "--frame-ms",
        type=int,
        default=100,
        help="Nominal Opus frame duration in ms (default: 100)",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds (0 = unlimited)",
    )
    p.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Stop after N packets (0 = unlimited)",
    )
    p.add_argument(
        "--append", action="store_true", help="Append to existing WAV if present"
    )
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    return p.parse_args(argv)


def open_wav(path: Path, channels: int, rate: int, append: bool):
    exists = path.exists()
    mode = "rb+" if append and exists else "wb"
    wf = wave.open(str(path), mode)
    if mode == "wb":
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
    else:
        if (
            wf.getnchannels() != channels
            or wf.getframerate() != rate
            or wf.getsampwidth() != 2
        ):
            raise ValueError("Existing WAV parameters don't match requested parameters")
        wf.seek(0, 2)
    return wf


def main(argv=None):  # noqa: C901
    args = parse_args(argv)
    port_path = args.port
    channels = args.channels
    sample_rate = args.rate
    frame_ms = args.frame_ms
    samples_per_frame = sample_rate * frame_ms // 1000

    decoder = Decoder(sample_rate, channels)

    try:
        ser = serial.Serial(port_path, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open serial port {port_path}: {e}", file=sys.stderr)
        return 2

    wav_path = Path(args.outfile)
    try:
        wav_file = open_wav(wav_path, channels, sample_rate, args.append)
    except Exception as e:
        print(f"Failed to open WAV file: {e}", file=sys.stderr)
        ser.close()
        return 3

    print(f"Recording from {port_path} -> {wav_path} (Ctrl+C to stop)")
    print(f"Frame: {frame_ms} ms, samples/frame: {samples_per_frame}")

    total_samples = 0
    packet_count = 0

    try:
        while True:
            line = ser.readline()
            if not line:
                continue
            line = line.strip()
            if not line:
                continue

            if not line.startswith(b"OPUS:"):
                print(line.decode(encoding="utf-8"))
                continue

            sep = line.find(b":")
            if sep != -1:
                line = line[sep + 1 :].strip()

            try:
                packet_bytes = base64.b64decode(line, validate=True)
            except Exception:
                if not args.quiet:
                    print(f"Skipping invalid base64 line: {line[:64]!r}")
                continue

            try:
                pcm = decoder.decode(packet_bytes, samples_per_frame, decode_fec=False)
            except Exception as e:
                if not args.quiet:
                    print(f"Opus decode error: {e}")
                continue

            frame = np.frombuffer(pcm, dtype=np.int16)
            wav_file.writeframes(frame.tobytes())
            total_samples += frame.size // channels
            packet_count += 1

            if not args.quiet and packet_count % 10 == 0:
                seconds = total_samples / sample_rate
                print(
                    f"Packets: {packet_count}, seconds: {seconds:.2f}, total samples: {total_samples}"
                )

            if args.max_packets and packet_count >= args.max_packets:
                print("Reached max packets limit")
                break
            if args.max_seconds and (total_samples / sample_rate) >= args.max_seconds:
                print("Reached max seconds limit")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        wav_file.close()
        ser.close()
        print("Saved WAV:", wav_path)
        print(
            f"Total packets: {packet_count}, duration ~ {total_samples / sample_rate:.2f}s"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
