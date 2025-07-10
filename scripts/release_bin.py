#! /usr/bin/env python3

# pylint: disable=missing-module-docstring
import argparse
import os
import subprocess
from datetime import datetime


# pylint: disable=missing-function-docstring
def get_current_dir_name():
    cwd = os.getcwd()
    name = os.path.basename(cwd)
    if not name:
        raise ValueError("Current directory name is empty.")
    return name


def get_current_project_name():
    cwd = os.getcwd()
    cmakelists_path = os.path.join(cwd, "CMakeLists.txt")
    if not os.path.exists(cmakelists_path):
        raise FileNotFoundError("CMakeLists.txt not found in the current directory.")
    with open(cmakelists_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("project("):
                return line.split("(")[1].split(")")[0].strip()
    raise ValueError("Project name not found in CMakeLists.txt.")


def get_current_date():
    return datetime.now().strftime("%y%m%d")


def get_git_commit_hash():
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()[:7]
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git commit hash: {e.stderr.strip()}") from e


def get_binary_hash(binary_path):
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary file {binary_path} does not exist.")
    res = subprocess.run(
        ["sha256sum", binary_path], capture_output=True, text=True, check=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"Error calculating hash for {binary_path}: {res.stderr}")
    return res.stdout.split()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Release binary for ESP32-S3.")
    parser.add_argument(
        "--chip", type=str, default="esp32s3", help="Chip type (default: esp32s3)"
    )
    parser.add_argument(
        "--flash_size", type=str, default="8MB", help="Flash size (default: 8MB)"
    )
    parser.add_argument(
        "--flash_freq", type=str, default="80m", help="Flash frequency (default: 80m)"
    )
    args = parser.parse_args()

    dir_name = get_current_dir_name()
    project_name = get_current_project_name()
    current_date = get_current_date()
    git_commit_hash = get_git_commit_hash()
    bin_name = f"{args.chip}_{dir_name}_{current_date}_{git_commit_hash}.bin"  # pylint: disable=invalid-name
    save_path = os.path.join("build", bin_name)

    command = [
        "python3",
        "-m",
        "esptool",
        "--chip",
        args.chip,
        "merge_bin",
        "-o",
        save_path,
        "--flash_size",
        args.flash_size,
        "--flash_freq",
        args.flash_freq,
        "0x0",
        "build/bootloader/bootloader.bin",
        "0x8000",
        "build/partition_table/partition-table.bin",
        "0x10000",
        f"build/{project_name}.bin",
    ]

    result = subprocess.run(command, check=True)
    if result.returncode != 0:
        raise RuntimeError("Failed to merge binary files.")

    print(f"Binary saved to {save_path}")
    print(f"SHA256 Hash: {get_binary_hash(save_path)}")
