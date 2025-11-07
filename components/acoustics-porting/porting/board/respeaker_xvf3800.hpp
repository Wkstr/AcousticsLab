#pragma once
#ifndef BOARD_RESPEAKER_XVF3800_HPP
#define BOARD_RESPEAKER_XVF3800_HPP

// Board: Seeed Studio ReSpeaker XVF3800
// Microphone: I2S XMOS (Master Mode)
// Sample Rate: 16kHz

#include <driver/gpio.h>
#include <driver/i2s_std.h>

namespace board { namespace respeaker_xvf3800 {

    // I2S Configuration
    constexpr bool USE_PDM_MODE = false;
    constexpr bool USE_I2S_STD_MODE = true;

    // I2S Role
    constexpr i2s_role_t I2S_ROLE = I2S_ROLE_MASTER;

    // I2S Standard Mode Pins
    constexpr gpio_num_t I2S_BCLK_PIN = GPIO_NUM_8;
    constexpr gpio_num_t I2S_WS_PIN = GPIO_NUM_7;
    constexpr gpio_num_t I2S_DIN_PIN = GPIO_NUM_43;
    constexpr gpio_num_t I2S_DOUT_PIN = GPIO_NUM_44;

    // Sample Rate
    constexpr uint32_t RAW_SAMPLE_RATE = 16000;

    // DMA Configuration
    constexpr uint32_t DMA_FRAME_COUNT = 800;

}} // namespace board::respeaker_xvf3800

#endif // BOARD_RESPEAKER_XVF3800_HPP
