#pragma once
#ifndef BOARD_XIAO_S3_HPP
#define BOARD_XIAO_S3_HPP

// Board: Seeed Studio XIAO ESP32S3
// Microphone: PDM Microphone
// Sample Rate: 44.1kHz

#include <driver/gpio.h>

namespace board { namespace xiao_s3 {

    // I2S Configuration
    constexpr bool USE_PDM_MODE = true;
    constexpr bool USE_I2S_STD_MODE = false;

    // PDM Microphone Pins
    constexpr gpio_num_t PDM_CLK_PIN = GPIO_NUM_42;
    constexpr gpio_num_t PDM_DIN_PIN = GPIO_NUM_41;

    // Sample Rate
    constexpr uint32_t RAW_SAMPLE_RATE = 44100;

    // DMA Configuration
    constexpr uint32_t DMA_FRAME_COUNT = 980;

}} // namespace board::xiao_s3

#endif // BOARD_XIAO_S3_HPP
