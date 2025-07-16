#pragma once
#ifndef MICROPHONE_HPP
#define MICROPHONE_HPP

#include "core/logger.hpp"
#include "core/ring_buffer.hpp"
#include "hal/sensor.hpp"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>
#include <microphone.h>

#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace porting {

using namespace hal;

static core::ConfigObjectMap DEFAULT_CONFIGS()
{
    core::ConfigObjectMap configs;

    configs.emplace("clk_pin", core::ConfigObject::createInteger("clk_pin", "I2S clock pin", 42, 0, 48));
    configs.emplace("data_pin", core::ConfigObject::createInteger("data_pin", "I2S data pin", 41, 0, 48));
    configs.emplace("i2s_port", core::ConfigObject::createInteger("i2s_port", "I2S port number", 0, 0, 1));
    configs.emplace("sample_rate",
        core::ConfigObject::createInteger("sample_rate", "Audio sample rate in Hz", 16000, 8000, 48000));
    configs.emplace("bit_width", core::ConfigObject::createInteger("bit_width", "Audio bit width", 16, 16, 32));
    configs.emplace("channel_mode",
        core::ConfigObject::createInteger("channel_mode", "Channel mode (0=mono, 1=stereo)", 0, 0, 1));
    configs.emplace("sampling_rate",
        core::ConfigObject::createInteger("sampling_rate", "Data sampling rate in Hz", 1000, 10, 1000));

    return configs;
}

/**
 * @brief Microphone sensor implementation following LIS3DHTR architecture pattern
 *
 * This class provides a HAL-compliant microphone sensor interface using I2S PDM.
 * It implements the same design patterns as SensorLIS3DHTR for consistency.
 */
class SensorMicrophone final: public Sensor
{
public:
    /**
     * @brief Constructor - registers sensor with default configuration
     */
    SensorMicrophone() : Sensor(Info(2, "I2S PDM Microphone", Type::Microphone, DEFAULT_CONFIGS())) { }

    /**
     * @brief Initialize the microphone sensor
     *
     * @return core::Status indicating success or failure
     */
    core::Status init() override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Sensor is already initialized or in an invalid state");
        }

        // Initialize microphone driver
        {
            int clk_pin = _info.configs["clk_pin"].getValue<int>();
            int data_pin = _info.configs["data_pin"].getValue<int>();
            int i2s_port = _info.configs["i2s_port"].getValue<int>();
            int sample_rate = _info.configs["sample_rate"].getValue<int>();
            int bit_width = _info.configs["bit_width"].getValue<int>();
            int channel_mode = _info.configs["channel_mode"].getValue<int>();

            if (!_microphone)
            {
                _microphone = new driver::Microphone();
            }

            auto ret = _microphone->init(static_cast<gpio_num_t>(clk_pin), static_cast<gpio_num_t>(data_pin),
                static_cast<i2s_port_t>(i2s_port), static_cast<driver::microphone_sample_rate_t>(sample_rate),
                static_cast<driver::microphone_bit_width_t>(bit_width),
                static_cast<driver::microphone_channel_mode_t>(channel_mode));

            if (ret != 0 && ret != EALREADY)
            {
                LOG(ERROR, "Failed to initialize microphone: %s", std::strerror(ret));
                delete _microphone;
                _microphone = nullptr;
                return STATUS_CODE(ret);
            }

            // Start audio capture
            ret = _microphone->startCapture();
            if (ret != 0)
            {
                LOG(ERROR, "Failed to start microphone capture: %s", std::strerror(ret));
                delete _microphone;
                _microphone = nullptr;
                return STATUS_CODE(ret);
            }
        }

        // Initialize ring buffer (first level cache)
        {
            _buffer = core::RingBuffer<AudioData>::create(_buffer_capacity);
            if (!_buffer)
            {
                LOG(ERROR, "Failed to allocate ring buffer of capacity: %zu", _buffer_capacity);
                return STATUS(ENOMEM, "Failed to allocate ring buffer");
            }
            LOG(DEBUG, "Allocated ring buffer of capacity: %zu at %p", _buffer_capacity, _buffer.get());
        }

        // Create audio capture task (like main repository pattern)
        {
            if (_capture_task_handle != nullptr)
            {
                LOG(WARNING, "Capture task already exists, terminating old task");
                vTaskDelete(_capture_task_handle);
                _capture_task_handle = nullptr;
            }

            _this = this;
            BaseType_t result = xTaskCreate(captureTask, // Task function
                "AudioCapture",                          // Task name
                4 * 1024,                                // Stack size (4KB like main repo)
                this,                                    // Task parameter (this pointer)
                10,                                      // Priority (same as main repo)
                &_capture_task_handle                    // Task handle
            );

            if (result != pdPASS)
            {
                LOG(ERROR, "Failed to create audio capture task");
                return STATUS(EFAULT, "Failed to create audio capture task");
            }

            _info.status = Status::Idle;
            LOG(DEBUG, "Audio capture task created successfully");
        }

        // Initialize data buffer (second level cache)
        {
            _frame_index = 0;
            if (!_data_buffer)
            {
                static_assert(_data_buffer_capacity >= 1, "Data buffer capacity must be at least 1");
                _data_buffer = std::make_shared<std::byte[]>(_data_buffer_capacity * sizeof(AudioData));
                if (!_data_buffer)
                {
                    LOG(ERROR, "Failed to allocate data buffer of size: %zu bytes",
                        _data_buffer_capacity * sizeof(AudioData));
                    return STATUS(ENOMEM, "Failed to allocate data buffer");
                }
                LOG(DEBUG, "Allocated data buffer of size: %zu bytes at %p", _data_buffer_capacity * sizeof(AudioData),
                    _data_buffer.get());
            }
        }

        // Initialize sensor status
        {
            _sensor_status = 0;
        }

        return STATUS_OK();
    }

    /**
     * @brief Deinitialize the microphone sensor
     *
     * @return core::Status indicating success or failure
     */
    core::Status deinit() override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status == Status::Uninitialized)
        {
            return STATUS_OK();
        }

        // Stop and delete capture task
        if (_capture_task_handle)
        {
            vTaskDelete(_capture_task_handle);
            _capture_task_handle = nullptr;
        }

        // Clear buffers
        if (_buffer)
        {
            _buffer.reset();
        }
        if (_data_buffer)
        {
            _data_buffer.reset();
        }

        // Deinitialize microphone driver
        if (_microphone)
        {
            _microphone->deinit();
            delete _microphone;
            _microphone = nullptr;
        }

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    /**
     * @brief Update sensor configuration (not supported)
     *
     * @param configs Configuration map
     * @return core::Status indicating operation not supported
     */
    core::Status updateConfig(const core::ConfigMap &configs) override
    {
        return STATUS(ENOTSUP, "Update config is not supported for microphone sensor");
    }

    /**
     * @brief Get number of available audio samples
     *
     * @return Number of available samples
     */
    inline size_t dataAvailable() const noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!_buffer)
        {
            LOG(ERROR, "Buffer is not initialized");
            return 0;
        }
        return _buffer->size() > _data_buffer_capacity ? _data_buffer_capacity : _buffer->size();
    }

    /**
     * @brief Clear all buffered data
     *
     * @return Number of samples cleared
     */
    inline size_t dataClear() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!_buffer)
        {
            LOG(ERROR, "Buffer is not initialized");
            return 0;
        }

        const size_t cleared = _buffer->size();
        _buffer->clear();
        LOG(DEBUG, "Cleared %zu audio samples from buffer", cleared);
        return cleared;
    }

    /**
     * @brief Read audio data frame
     *
     * @param data_frame Output data frame containing audio tensor
     * @param batch_size Number of samples to read
     * @return core::Status indicating success or failure
     */
    inline core::Status readDataFrame(core::DataFrame<core::Tensor> &data_frame, size_t batch_size) override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!initialized())
        {
            LOG(ERROR, "Sensor is not initialized");
            return STATUS(ENXIO, "Sensor is not initialized");
        }

        if (_sensor_status != 0)
        {
            LOG(ERROR, "Sensor status is invalid: %d", _sensor_status);
            return STATUS(_sensor_status, "Sensor status is invalid");
        }

        if (!_buffer)
        {
            LOG(ERROR, "Buffer is not initialized");
            return STATUS(EFAULT, "Buffer is not initialized");
        }

        if (batch_size > _buffer_capacity)
        {
            LOG(ERROR, "Batch size exceeds buffer capacity: %zu > %zu", batch_size, _buffer_capacity);
            return STATUS(EOVERFLOW, "Batch size exceeds buffer capacity");
        }

        if (_buffer->size() < batch_size)
        {
            LOG(DEBUG, "Not enough data in buffer: %zu < %zu", _buffer->size(), batch_size);
            return STATUS(EAGAIN, "Not enough data in buffer");
        }

        if (!_data_buffer)
        {
            LOG(ERROR, "Data buffer is not initialized");
            return STATUS(EFAULT, "Data buffer is not initialized");
        }

        if (batch_size > _data_buffer_capacity)
        {
            LOG(ERROR, "Batch size exceeds data buffer capacity: %zu > %zu", batch_size, _data_buffer_capacity);
            return STATUS(EOVERFLOW, "Batch size exceeds data buffer capacity");
        }

        _info.status = Status::Locked;

        // Discard old data if buffer has more than requested
        const size_t n_elems_to_discard = _buffer->size() - batch_size;
        if (n_elems_to_discard > 0)
        {
            const auto discarded = _buffer->read(nullptr, n_elems_to_discard);
            LOG(DEBUG, "Discarded %zu staled elements from buffer", discarded);
        }

        // Create a new data buffer for this frame to avoid ownership conflicts
        auto frame_buffer = std::make_shared<std::byte[]>(batch_size * sizeof(AudioData));
        if (!frame_buffer)
        {
            LOG(ERROR, "Failed to allocate frame buffer");
            _info.status = Status::Idle;
            return STATUS(ENOMEM, "Failed to allocate frame buffer");
        }

        // Read data into frame buffer
        data_frame.timestamp = std::chrono::steady_clock::now();
        data_frame.index = _frame_index;
        _buffer->read(reinterpret_cast<AudioData *>(frame_buffer.get()), batch_size);

        // Create tensor with audio data (1D tensor for mono audio)
        data_frame.data
            = core::Tensor(core::Tensor::Type::Int16, { batch_size }, frame_buffer, batch_size * sizeof(AudioData));
        _frame_index += batch_size;

        _info.status = Status::Idle;

        return STATUS_OK();
    }

public:
    /**
     * @brief Audio data structure for single sample
     */
    struct __attribute__((packed)) AudioData
    {
        int16_t sample;
    };

protected:
    /**
     * @brief Audio capture task (like main repository pattern)
     * Continuously reads audio data from I2S and puts into ring buffer
     * @param pvParameters Task parameter (this pointer)
     */
    static void captureTask(void *pvParameters) noexcept
    {
        SensorMicrophone *sensor = static_cast<SensorMicrophone *>(pvParameters);
        if (!sensor || !sensor->_microphone || !sensor->_buffer)
        {
            LOG(ERROR, "Capture task called with invalid state");
            vTaskDelete(nullptr);
            return;
        }

        // Use larger buffer for continuous streaming (like main repo: 4410 samples)
        constexpr size_t CHUNK_SAMPLES = 4410; // 100ms at 44100Hz
        int16_t *chunk_buffer = static_cast<int16_t *>(malloc(CHUNK_SAMPLES * sizeof(int16_t)));
        if (!chunk_buffer)
        {
            LOG(ERROR, "Failed to allocate chunk buffer");
            vTaskDelete(nullptr);
            return;
        }

        LOG(INFO, "Audio capture task started");

        while (true)
        {
            size_t bytes_read = 0;
            int ret = sensor->_microphone->readAudioData(chunk_buffer, CHUNK_SAMPLES, bytes_read, 100); // 100ms timeout

            if (ret != 0)
            {
                LOG(ERROR, "Failed to read audio data: %d", ret);
                sensor->_sensor_status = ret;
                vTaskDelay(pdMS_TO_TICKS(10)); // Brief delay before retry
                continue;
            }

            // Convert to expected number of samples
            const size_t samples_read = bytes_read / sizeof(int16_t);
            if (samples_read == 0)
            {
                vTaskDelay(pdMS_TO_TICKS(1)); // Yield to other tasks
                continue;
            }

            // Push samples to ring buffer
            for (size_t i = 0; i < samples_read; ++i)
            {
                AudioData data = { chunk_buffer[i] };
                if (!sensor->_buffer->put(data))
                {
                    // Buffer full, skip oldest data (normal in streaming)
                    break;
                }
            }

            // Yield to other tasks to prevent watchdog timeout
            vTaskDelay(pdMS_TO_TICKS(1));
        }

        free(chunk_buffer);
        vTaskDelete(nullptr);
    }

private:
    // Thread safety
    mutable std::mutex _lock;

    // Microphone driver instance
    driver::Microphone *_microphone = nullptr;

    // Buffer configuration - Aligned with main repository requirements
    // Main uses 44032 samples (1 second) analysis window with 22016 samples (0.5 second) stride
    // AudioProvider delivers 4410 samples (100ms) per chunk
    static constexpr size_t _buffer_capacity = 48000;      // ~1.1 seconds at 44100Hz, enough for analysis window
    static constexpr size_t _data_buffer_capacity = 44032; // Match main's ANALYSIS_WINDOW_SAMPLES

    // First level cache (ring buffer)
    std::unique_ptr<core::RingBuffer<AudioData>> _buffer;

    // Second level cache (shared data buffer)
    std::shared_ptr<std::byte[]> _data_buffer = nullptr;

    // Status and control
    volatile int _sensor_status = 0;
    TaskHandle_t _capture_task_handle = nullptr;
    static SensorMicrophone *_this;

    // Frame tracking
    size_t _frame_index = 0;
};

// Static member definition
SensorMicrophone *SensorMicrophone::_this = nullptr;

} // namespace porting

#endif // MICROPHONE_HPP
