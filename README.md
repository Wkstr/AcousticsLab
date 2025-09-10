# AcousticsLab

AcousticsLab is a cross-platform framework for sound and vibration analysis, with advanced algorithms for anomaly detection and audio classification integrated and optimized, we deliver a simple, universal, and robust acoustics analysis toolkit for MCUs and SBCs.

## Getting Started

Please follow the instructions in [ESP-IDF - Get Started](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) to setup the build toolchain first. Currently we're using the latest version `v5.4.1`.

1. Clone and setup the repository.

    ```sh
    git clone https://github.com/Seeed-Studio/AcousticsLab.git --depth=1
    cd AcousticsLab
    git submodule update --init
    ```

2. Build and run examples (replace `DEMO` with the example you want to run).

    ```sh
    cd examples/${DEMO}
    idf.py set-target esp32s3
    idf.py build
    idf.py flash monitor
    ```

## Web Console

AcousticsLab implements a user-friendly web interface to visualize audio/vibration data, control the device, and view results in real-time.

<img src="https://github.com/Seeed-Studio/AcousticsLab/blob/main/docs/images/console-preview.gif?raw=true" alt="Console Preview" width="100%"/>

1. Install the required python dependencies.

    ```sh
    pip3 install -r requirements.txt
    ```

2. Launch console with the device connected to your computer via USB cable.

    ```sh
    python3 scripts/console.py --help # see available options
    python3 scripts/console.py        # launch console with default options
    ```

3. Open the console in your web browser at `http://localhost:7860`.

## Algorithms

AcousticsLab aims to provide a comprehensive set of SOTA algorithms for sound and vibration analysis. The current implementation includes:

- [Gyroscope Euclidean Distance Anomaly Detection (GEDAD)](docs/algorithms/gedad/how-it-works.md): A lightweight algorithm for detecting anomalies in gyroscope and 3-axis accelerometer data.

- [Speech Commands](https://arxiv.org/pdf/1804.03209): An algorithm for recognizing keywords and simple audio events, compatible with models trained using tools like Google's [Teachable Machine](https://teachablemachine.withgoogle.com/).

## Architecture Overview

This diagram shows a modular embedded AI architecture. It features a layered design with a Hardware Abstraction Layer (HAL), a Core Library (with DSP/ML), an API, and an Application layer. The architecture is designed to simplify development and integrate easily with platforms like Arduino and MicroPython.

<img src="https://github.com/Seeed-Studio/AcousticsLab/blob/main/docs/images/architecture-overview.png?raw=true" alt="Architecture Overview" width="100%"/>

The high-level architecture consists of 2 main components:
- [Acoustics](https://github.com/Seeed-Studio/AcousticsLab/tree/main/components/acoustics): features the core functionality for sound and vibration analysis, including algorithms for anomaly detection and audio classification.
- [Acoustics-Porting](https://github.com/Seeed-Studio/AcousticsLab/tree/main/components/acoustics-porting): provides the hardware abstraction layer (HAL) for the Acoustics component, allowing it to run on different hardware platforms.

## License

This software is licensed under the GPLv3 license, see [LICENSE](LICENSE) for more information.
