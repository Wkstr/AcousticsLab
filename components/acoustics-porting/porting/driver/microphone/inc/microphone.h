#pragma once
#ifndef MICROPHONE_H
#define MICROPHONE_H

#include <driver/i2s_pdm.h>
#include <driver/gpio.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cstddef>
#include <cstdint>

namespace driver {

/**
 * @brief Microphone sample rate configurations
 */
typedef enum {
    MICROPHONE_SAMPLE_RATE_8KHZ = 8000,
    MICROPHONE_SAMPLE_RATE_16KHZ = 16000,
    MICROPHONE_SAMPLE_RATE_22KHZ = 22050,
    MICROPHONE_SAMPLE_RATE_44KHZ = 44100,
    MICROPHONE_SAMPLE_RATE_48KHZ = 48000,
} microphone_sample_rate_t;

/**
 * @brief Microphone bit width configurations
 */
typedef enum {
    MICROPHONE_BIT_WIDTH_16 = I2S_DATA_BIT_WIDTH_16BIT,
    MICROPHONE_BIT_WIDTH_24 = I2S_DATA_BIT_WIDTH_24BIT,
    MICROPHONE_BIT_WIDTH_32 = I2S_DATA_BIT_WIDTH_32BIT,
} microphone_bit_width_t;

/**
 * @brief Microphone channel mode
 */
typedef enum {
    MICROPHONE_CHANNEL_MONO = I2S_SLOT_MODE_MONO,
    MICROPHONE_CHANNEL_STEREO = I2S_SLOT_MODE_STEREO,
} microphone_channel_mode_t;

/**
 * @brief Microphone driver class for I2S PDM interface
 * 
 * This class provides a hardware abstraction layer for PDM microphones
 * connected via I2S interface. It follows the same design pattern as
 * the LIS3DHTR driver for consistency within the AcousticsLab framework.
 */
class Microphone final
{
public:
    /**
     * @brief Constructor
     */
    Microphone();

    /**
     * @brief Destructor - automatically calls deinit()
     */
    ~Microphone();

    /**
     * @brief Initialize the microphone with specified configuration
     * 
     * @param clk_pin GPIO pin for I2S clock signal
     * @param data_pin GPIO pin for I2S data signal  
     * @param i2s_port I2S port number to use
     * @param sample_rate Audio sample rate in Hz
     * @param bit_width Audio bit width
     * @param channel_mode Mono or stereo channel mode
     * @param timeout_ms Timeout for initialization operations
     * @return 0 on success, error code on failure
     *         EALREADY if already initialized
     *         EIO if I2S configuration failed
     *         ENODEV if device probe failed
     */
    int init(gpio_num_t clk_pin, gpio_num_t data_pin, i2s_port_t i2s_port = I2S_NUM_0,
             microphone_sample_rate_t sample_rate = MICROPHONE_SAMPLE_RATE_44KHZ,
             microphone_bit_width_t bit_width = MICROPHONE_BIT_WIDTH_16,
             microphone_channel_mode_t channel_mode = MICROPHONE_CHANNEL_MONO,
             int timeout_ms = 1000);

    /**
     * @brief Deinitialize the microphone and free resources
     */
    void deinit();

    /**
     * @brief Start audio capture
     * 
     * @param timeout_ms Timeout for start operation
     * @return 0 on success, error code on failure
     *         ENODEV if not initialized
     *         EIO if start failed
     */
    int startCapture(int timeout_ms = 1000);

    /**
     * @brief Stop audio capture
     * 
     * @param timeout_ms Timeout for stop operation  
     * @return 0 on success, error code on failure
     *         ENODEV if not initialized
     *         EIO if stop failed
     */
    int stopCapture(int timeout_ms = 1000);

    /**
     * @brief Read audio data from microphone
     * 
     * @param buffer Buffer to store audio samples
     * @param samples Number of samples to read
     * @param bytes_read Actual number of bytes read
     * @param timeout_ms Timeout for read operation
     * @return 0 on success, error code on failure
     *         ENODEV if not initialized
     *         EIO if read failed
     *         ETIMEDOUT if timeout occurred
     */
    int readAudioData(int16_t* buffer, size_t samples, size_t& bytes_read, int timeout_ms = 1000);

    /**
     * @brief Get current device status
     * 
     * @param timeout_ms Timeout for status check
     * @return 0 if device is ready, error code otherwise
     *         ENODEV if not initialized
     *         EIO if device error
     */
    int getDeviceStatus(int timeout_ms = 1000);

    /**
     * @brief Check if microphone is initialized
     * 
     * @return true if initialized, false otherwise
     */
    inline bool isInitialized() const noexcept { return _initialized; }

    /**
     * @brief Check if microphone is currently capturing
     * 
     * @return true if capturing, false otherwise
     */
    inline bool isCapturing() const noexcept { return _capturing; }

private:
    // I2S channel configuration and handles
    i2s_chan_config_t _chan_config;
    i2s_chan_handle_t _rx_handle;
    i2s_pdm_rx_config_t _pdm_config;

    // Configuration parameters
    gpio_num_t _clk_pin;
    gpio_num_t _data_pin;
    i2s_port_t _i2s_port;
    microphone_sample_rate_t _sample_rate;
    microphone_bit_width_t _bit_width;
    microphone_channel_mode_t _channel_mode;

    // State management
    bool _initialized;
    bool _capturing;

    /**
     * @brief Internal helper to validate configuration parameters
     * 
     * @return true if configuration is valid, false otherwise
     */
    bool _validateConfig() const noexcept;

    /**
     * @brief Internal helper to setup I2S channel configuration
     */
    void _setupChannelConfig();

    /**
     * @brief Internal helper to setup PDM RX configuration
     */
    void _setupPdmConfig();
};

} // namespace driver

#endif // MICROPHONE_H
