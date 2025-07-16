#include "microphone.h"

#include <esp_check.h>
#include <esp_log.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <cerrno>
#include <cstring>

static const char *TAG = "Microphone";

// Conversion delay similar to LIS3DHTR
#define MICROPHONE_INIT_DELAY_MS (10)

namespace driver {

Microphone::Microphone()
{
    _chan_config = {};
    _rx_handle = nullptr;
    _pdm_config = {};

    _clk_pin = GPIO_NUM_NC;
    _data_pin = GPIO_NUM_NC;
    _i2s_port = I2S_NUM_0;
    _sample_rate = MICROPHONE_SAMPLE_RATE_44KHZ;
    _bit_width = MICROPHONE_BIT_WIDTH_16;
    _channel_mode = MICROPHONE_CHANNEL_MONO;

    _initialized = false;
    _capturing = false;
}

Microphone::~Microphone()
{
    deinit();
}

int Microphone::init(gpio_num_t clk_pin, gpio_num_t data_pin, i2s_port_t i2s_port, microphone_sample_rate_t sample_rate,
    microphone_bit_width_t bit_width, microphone_channel_mode_t channel_mode, int timeout_ms)
{
    // Check if already initialized (similar to LIS3DHTR pattern)
    if (_initialized && _rx_handle)
    {
        ESP_LOGW(TAG, "Microphone already initialized");
        return EALREADY;
    }

    // Store configuration parameters
    _clk_pin = clk_pin;
    _data_pin = data_pin;
    _i2s_port = i2s_port;
    _sample_rate = sample_rate;
    _bit_width = bit_width;
    _channel_mode = channel_mode;

    // Validate configuration
    if (!_validateConfig())
    {
        ESP_LOGE(TAG, "Invalid configuration parameters");
        return EINVAL;
    }

    // Setup I2S channel configuration
    _setupChannelConfig();

    // Create I2S channel
    esp_err_t ret = i2s_new_channel(&_chan_config, NULL, &_rx_handle);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to create I2S channel: %s", esp_err_to_name(ret));
        deinit();
        return EIO;
    }

    // Setup PDM RX configuration
    _setupPdmConfig();

    // Initialize PDM RX mode
    ret = i2s_channel_init_pdm_rx_mode(_rx_handle, &_pdm_config);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to initialize PDM RX mode: %s", esp_err_to_name(ret));
        deinit();
        return EIO;
    }

    // Enable I2S channel
    ret = i2s_channel_enable(_rx_handle);
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to enable I2S channel: %s", esp_err_to_name(ret));
        deinit();
        return EIO;
    }

    // Add initialization delay (similar to LIS3DHTR pattern)
    vTaskDelay(pdMS_TO_TICKS(MICROPHONE_INIT_DELAY_MS));

    _initialized = true;
    ESP_LOGI(TAG, "Microphone initialized successfully (CLK: %d, DATA: %d, Rate: %d Hz)", _clk_pin, _data_pin,
        _sample_rate);

    return 0;
}

void Microphone::deinit()
{
    if (_capturing)
    {
        stopCapture();
    }

    if (_rx_handle)
    {
        i2s_channel_disable(_rx_handle);
        i2s_del_channel(_rx_handle);
        _rx_handle = nullptr;
    }

    _initialized = false;
    _capturing = false;

    ESP_LOGD(TAG, "Microphone deinitialized");
}

int Microphone::startCapture(int timeout_ms)
{
    if (!_initialized || !_rx_handle)
    {
        ESP_LOGE(TAG, "Microphone not initialized");
        return ENODEV;
    }

    if (_capturing)
    {
        ESP_LOGW(TAG, "Capture already started");
        return 0; // Not an error, just already started
    }

    // I2S channel is already enabled during init, just mark as capturing
    _capturing = true;
    ESP_LOGD(TAG, "Audio capture started");

    return 0;
}

int Microphone::stopCapture(int timeout_ms)
{
    if (!_initialized || !_rx_handle)
    {
        ESP_LOGE(TAG, "Microphone not initialized");
        return ENODEV;
    }

    if (!_capturing)
    {
        ESP_LOGW(TAG, "Capture already stopped");
        return 0; // Not an error, just already stopped
    }

    _capturing = false;
    ESP_LOGD(TAG, "Audio capture stopped");

    return 0;
}

int Microphone::readAudioData(int16_t *buffer, size_t samples, size_t &bytes_read, int timeout_ms)
{
    if (!_initialized || !_rx_handle)
    {
        ESP_LOGE(TAG, "Microphone not initialized");
        return ENODEV;
    }

    if (!buffer || samples == 0)
    {
        ESP_LOGE(TAG, "Invalid buffer or sample count");
        return EINVAL;
    }

    size_t bytes_to_read = samples * sizeof(int16_t);
    esp_err_t ret = i2s_channel_read(_rx_handle, buffer, bytes_to_read, &bytes_read, pdMS_TO_TICKS(timeout_ms));

    if (ret == ESP_ERR_TIMEOUT)
    {
        ESP_LOGW(TAG, "Read timeout occurred");
        return ETIMEDOUT;
    }
    else if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "Failed to read audio data: %s", esp_err_to_name(ret));
        return EIO;
    }

    return 0;
}

int Microphone::getDeviceStatus(int timeout_ms)
{
    if (!_initialized || !_rx_handle)
    {
        return ENODEV;
    }

    // For I2S PDM, if channel is created and enabled, device is considered ready
    // Additional status checks could be added here if needed
    return 0;
}

bool Microphone::_validateConfig() const noexcept
{
    // Validate GPIO pins
    if (_clk_pin == GPIO_NUM_NC || _data_pin == GPIO_NUM_NC)
    {
        ESP_LOGE(TAG, "Invalid GPIO pins: CLK=%d, DATA=%d", _clk_pin, _data_pin);
        return false;
    }

    // Validate I2S port (ESP32-S3 has I2S_NUM_0 and I2S_NUM_1)
    if (_i2s_port < I2S_NUM_0 || _i2s_port > I2S_NUM_1)
    {
        ESP_LOGE(TAG, "Invalid I2S port: %d", _i2s_port);
        return false;
    }

    // Validate sample rate
    if (_sample_rate < 8000 || _sample_rate > 48000)
    {
        ESP_LOGE(TAG, "Invalid sample rate: %d", _sample_rate);
        return false;
    }

    return true;
}

void Microphone::_setupChannelConfig()
{
    _chan_config = I2S_CHANNEL_DEFAULT_CONFIG(_i2s_port, I2S_ROLE_MASTER);
}

void Microphone::_setupPdmConfig()
{
    _pdm_config = {
        .clk_cfg = I2S_PDM_RX_CLK_DEFAULT_CONFIG(static_cast<uint32_t>(_sample_rate)),
        .slot_cfg = I2S_PDM_RX_SLOT_DEFAULT_CONFIG(static_cast<i2s_data_bit_width_t>(_bit_width), 
                                                   static_cast<i2s_slot_mode_t>(_channel_mode)),
        .gpio_cfg = {
            .clk = _clk_pin,
            .din = _data_pin,
            .invert_flags = {
                .clk_inv = false,
            },
        },
    };
}

} // namespace driver
