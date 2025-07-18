#pragma once
#ifndef MICROPHONE_HPP
#define MICROPHONE_HPP

#include "core/logger.hpp"
#include "core/ring_buffer.hpp"
#include "hal/sensor.hpp"

#include <driver/gpio.h>
#include <driver/i2s_pdm.h>
#include <esp_err.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include <atomic>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace porting {

using namespace hal;

// Microphone configuration types (moved from driver layer)
typedef enum {
    MICROPHONE_SAMPLE_RATE_8KHZ = 8000,
    MICROPHONE_SAMPLE_RATE_16KHZ = 16000,
    MICROPHONE_SAMPLE_RATE_22KHZ = 22050,
    MICROPHONE_SAMPLE_RATE_44KHZ = 44100,
    MICROPHONE_SAMPLE_RATE_48KHZ = 48000,
} microphone_sample_rate_t;

typedef enum {
    MICROPHONE_BIT_WIDTH_16 = I2S_DATA_BIT_WIDTH_16BIT,
    MICROPHONE_BIT_WIDTH_24 = I2S_DATA_BIT_WIDTH_24BIT,
    MICROPHONE_BIT_WIDTH_32 = I2S_DATA_BIT_WIDTH_32BIT,
} microphone_bit_width_t;

typedef enum {
    MICROPHONE_CHANNEL_MONO = I2S_SLOT_MODE_MONO,
    MICROPHONE_CHANNEL_STEREO = I2S_SLOT_MODE_STEREO,
} microphone_channel_mode_t;

// Initialization delay constant
#define MICROPHONE_INIT_DELAY_MS (10)

// Audio buffer configuration - based on ESP I2S DMA design
// ESP I2S DMA: 6 slots, each 50ms (2205 frames at 44.1kHz)
// Ring buffer: 16 slots for 800ms total buffering
static constexpr size_t DMA_BUFFER_SLOTS = 6;        // ESP I2S DMA buffer slots
static constexpr size_t DMA_FRAMES_PER_SLOT = 2205;  // 50ms at 44.1kHz (sr * 1/freq = 44100 * 1/20)
static constexpr size_t RING_BUFFER_SLOTS = 16;      // Ring buffer slots (800ms total)
static constexpr size_t FRAMES_PER_RB_SLOT = 2205;   // Same as DMA slot for simplicity
static constexpr size_t FRAME_SIZE = 44032;          // Feature extraction frame (1 second)
static constexpr size_t FRONTEND_BUFFER_SIZE = 4410; // Frontend buffer (100ms)

// IRAM-safe constants
#define IRAM_RING_BUFFER_SLOTS    16
#define IRAM_FRAMES_PER_RB_SLOT   2205
#define IRAM_NORMALIZATION_FACTOR (1.0f / 32768.0f)

/**
 * @brief Ring buffer slot structure
 * rb slot { ts, frames <- dma buf slot frames }
 */
struct RingBufferSlot
{
    std::chrono::steady_clock::time_point timestamp;
    std::vector<float> frames;

    RingBufferSlot() : frames(FRAMES_PER_RB_SLOT) { }
};

/**
 * @brief Audio buffer system based on ESP I2S DMA design
 *
 * ESP I2S DMA: 6 slots, each 50ms
 * Ring buffer: 16 slots for 800ms total
 * I2S callback: rb.push <- slot
 * High level: microphone.getData() -> rb.pop
 */
struct AudioBufferSystem
{
    // Ring buffer: rb <- [[slot], ...]
    std::vector<RingBufferSlot> ring_buffer;
    std::atomic<int> write_index { 0 }; // For rb.push
    std::atomic<int> read_index { 0 };  // For rb.pop

    // Frontend buffer for non-overlapping data
    std::vector<float> frontend_buffer;
    std::atomic<size_t> frontend_write_pos { 0 };
    std::atomic<size_t> frontend_read_pos { 0 };

    // Feature extraction buffer for overlapping data
    std::vector<float> tensor_buffer;
    std::atomic<int> tensor_read_index { 0 };

    // Synchronization for non-IRAM operations
    mutable std::mutex buffer_mutex;

    AudioBufferSystem()
        : ring_buffer(RING_BUFFER_SLOTS), frontend_buffer(FRONTEND_BUFFER_SIZE), tensor_buffer(FRAME_SIZE)
    {
    }
};

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

        // Initialize I2S PDM directly (integrated from driver layer)
        {
            int clk_pin = _info.configs["clk_pin"].getValue<int>();
            int data_pin = _info.configs["data_pin"].getValue<int>();
            int i2s_port = _info.configs["i2s_port"].getValue<int>();
            int sample_rate = _info.configs["sample_rate"].getValue<int>();
            int bit_width = _info.configs["bit_width"].getValue<int>();
            int channel_mode = _info.configs["channel_mode"].getValue<int>();

            // Store configuration
            _clk_pin = static_cast<gpio_num_t>(clk_pin);
            _data_pin = static_cast<gpio_num_t>(data_pin);
            _i2s_port = static_cast<i2s_port_t>(i2s_port);
            _sample_rate = static_cast<microphone_sample_rate_t>(sample_rate);
            _bit_width = static_cast<microphone_bit_width_t>(bit_width);
            _channel_mode = static_cast<microphone_channel_mode_t>(channel_mode);

            // Validate configuration
            if (!_validateConfig())
            {
                LOG(ERROR, "Invalid I2S configuration parameters");
                return STATUS(EINVAL, "Invalid I2S configuration parameters");
            }

            // Setup I2S channel configuration
            _setupChannelConfig();

            // Create I2S channel
            esp_err_t ret = i2s_new_channel(&_chan_config, NULL, &_rx_handle);
            if (ret != ESP_OK)
            {
                LOG(ERROR, "Failed to create I2S channel: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to create I2S channel");
            }

            // Setup PDM RX configuration
            _setupPdmConfig();

            // Initialize PDM RX mode
            ret = i2s_channel_init_pdm_rx_mode(_rx_handle, &_pdm_config);
            if (ret != ESP_OK)
            {
                LOG(ERROR, "Failed to initialize PDM RX mode: %s", esp_err_to_name(ret));
                _cleanupI2S();
                return STATUS(EIO, "Failed to initialize PDM RX mode");
            }

            // Setup DMA callbacks BEFORE enabling the channel
            ret = _setupDmaCallbacks();
            if (ret != ESP_OK)
            {
                LOG(ERROR, "Failed to setup DMA callbacks: %s", esp_err_to_name(ret));
                _cleanupI2S();
                return STATUS(EIO, "Failed to setup DMA callbacks");
            }

            // Enable I2S channel AFTER setting up callbacks
            ret = i2s_channel_enable(_rx_handle);
            if (ret != ESP_OK)
            {
                LOG(ERROR, "Failed to enable I2S channel: %s", esp_err_to_name(ret));
                _cleanupI2S();
                return STATUS(EIO, "Failed to enable I2S channel");
            }

            // Initialize audio buffer system
            _audio_buffer = std::make_unique<AudioBufferSystem>();
            _dma_buffer.resize(DMA_BUFFER_SIZE);

            // Add initialization delay
            vTaskDelay(pdMS_TO_TICKS(MICROPHONE_INIT_DELAY_MS));

            _i2s_initialized = true;
            LOG(INFO, "I2S PDM microphone with DMA callbacks initialized successfully (CLK: %d, DATA: %d, Rate: %d Hz)",
                _clk_pin, _data_pin, _sample_rate);
            LOG(INFO, "Ring buffer: %d slots, %d frames/slot, total %d frames (%d ms)", RING_BUFFER_SLOTS,
                FRAMES_PER_RB_SLOT, RING_BUFFER_SLOTS * FRAMES_PER_RB_SLOT,
                (RING_BUFFER_SLOTS * FRAMES_PER_RB_SLOT * 1000) / _sample_rate);
        }

        // DMA-based audio capture is now handled by I2S callbacks
        // No need for separate capture task or ring buffer
        {
            _frame_index = 0;
            _info.status = Status::Idle;
            LOG(INFO, "Audio capture using DMA callbacks is ready");
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

        // Clear audio buffer system
        if (_audio_buffer)
        {
            _audio_buffer.reset();
        }

        // Cleanup I2S resources
        _cleanupI2S();

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

        if (!_audio_buffer)
        {
            LOG(ERROR, "Audio buffer system is not initialized");
            return 0;
        }

        // Return available slots in ring buffer
        return _getAvailableBufferSlots() * FRAMES_PER_RB_SLOT;
    }

    /**
     * @brief Clear all buffered data
     *
     * @return Number of samples cleared
     */
    inline size_t dataClear() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (!_audio_buffer)
        {
            LOG(ERROR, "Audio buffer system is not initialized");
            return 0;
        }

        const size_t cleared = _getAvailableBufferSlots() * FRAMES_PER_RB_SLOT;
        _resetRingBuffer();
        LOG(DEBUG, "Cleared %zu audio samples from ring buffer", cleared);
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

        // Process any pending DMA data first
        _processDmaData();

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

        if (!_audio_buffer)
        {
            LOG(ERROR, "Audio buffer system is not initialized");
            return STATUS(EFAULT, "Audio buffer system is not initialized");
        }

        // For feature extraction, use overlapping frames
        // For regular data reading, use frontend buffer
        if (batch_size == FRAME_SIZE)
        {
            // This is likely a feature extraction request - use overlapping frames
            core::Tensor tensor = _getOverlappingFrame();
            if (tensor.dataSize() == 0)
            {
                LOG(DEBUG, "Not enough data for overlapping frame");
                return STATUS(EAGAIN, "Not enough data for overlapping frame");
            }

            data_frame.timestamp = std::chrono::steady_clock::now();
            data_frame.index = _frame_index++;
            data_frame.data = std::move(tensor);

            _info.status = Status::Idle;
            return STATUS_OK();
        }
        else
        {
            // Regular data reading - use frontend buffer
            if (batch_size > FRONTEND_BUFFER_SIZE)
            {
                LOG(ERROR, "Batch size exceeds frontend buffer capacity: %zu > %zu", batch_size, FRONTEND_BUFFER_SIZE);
                return STATUS(EOVERFLOW, "Batch size exceeds frontend buffer capacity");
            }
        }

        _info.status = Status::Locked;

        // Create a buffer for frontend data
        auto frame_buffer = std::make_shared<float[]>(batch_size);
        if (!frame_buffer)
        {
            LOG(ERROR, "Failed to allocate frame buffer");
            _info.status = Status::Idle;
            return STATUS(ENOMEM, "Failed to allocate frame buffer");
        }

        // Read data from frontend buffer
        size_t samples_read = _getFrontendData(frame_buffer.get(), batch_size);
        if (samples_read < batch_size)
        {
            LOG(DEBUG, "Not enough data in frontend buffer: %zu < %zu", samples_read, batch_size);
            _info.status = Status::Idle;
            return STATUS(EAGAIN, "Not enough data in frontend buffer");
        }

        // Convert float data to int16_t for compatibility
        auto int16_buffer = std::make_shared<std::byte[]>(batch_size * sizeof(AudioData));
        AudioData *audio_data = reinterpret_cast<AudioData *>(int16_buffer.get());

        for (size_t i = 0; i < batch_size; ++i)
        {
            // Convert float [-1.0, 1.0] back to int16_t
            audio_data[i].sample = static_cast<int16_t>(frame_buffer[i] * 32767.0f);
        }

        // Create tensor with audio data (1D tensor for mono audio)
        data_frame.timestamp = std::chrono::steady_clock::now();
        data_frame.index = _frame_index++;
        data_frame.data
            = core::Tensor(core::Tensor::Type::Int16, { batch_size }, int16_buffer, batch_size * sizeof(AudioData));

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
    // Audio capture is now handled by I2S DMA callbacks
    // No separate capture task needed

private:
    // Thread safety
    mutable std::mutex _lock;

    // I2S configuration and handles (integrated from driver layer)
    i2s_chan_config_t _chan_config = {};
    i2s_chan_handle_t _rx_handle = nullptr;
    i2s_pdm_rx_config_t _pdm_config = {};

    // Configuration parameters
    gpio_num_t _clk_pin = GPIO_NUM_NC;
    gpio_num_t _data_pin = GPIO_NUM_NC;
    i2s_port_t _i2s_port = I2S_NUM_0;
    microphone_sample_rate_t _sample_rate = MICROPHONE_SAMPLE_RATE_44KHZ;
    microphone_bit_width_t _bit_width = MICROPHONE_BIT_WIDTH_16;
    microphone_channel_mode_t _channel_mode = MICROPHONE_CHANNEL_MONO;

    // State management
    bool _i2s_initialized = false;

    // Audio buffer system (replaces ring buffer)
    std::unique_ptr<AudioBufferSystem> _audio_buffer;

    // DMA buffer for I2S callbacks
    static constexpr size_t DMA_BUFFER_SIZE = FRAMES_PER_RB_SLOT * sizeof(int16_t);
    std::vector<uint8_t> _dma_buffer;

    // Status and control
    volatile int _sensor_status = 0;
    TaskHandle_t _capture_task_handle = nullptr;
    static SensorMicrophone *_this;

    // Frame tracking
    size_t _frame_index = 0;

    // DMA callback data (for IRAM-safe processing)
    volatile void *_dma_data_ptr = nullptr;
    volatile size_t _dma_data_size = 0;
    std::atomic<bool> _new_data_flag { false };

    // Debug counters
    std::atomic<uint32_t> _dma_callback_count { 0 };
    uint32_t _data_process_count = 0;
    uint32_t _last_logged_callback_count = 0;

    /**
     * @brief Validate I2S configuration parameters
     */
    bool _validateConfig() const noexcept
    {
        // Validate GPIO pins
        if (_clk_pin == GPIO_NUM_NC || _data_pin == GPIO_NUM_NC)
        {
            return false;
        }

        // Validate sample rate
        switch (_sample_rate)
        {
            case MICROPHONE_SAMPLE_RATE_8KHZ:
            case MICROPHONE_SAMPLE_RATE_16KHZ:
            case MICROPHONE_SAMPLE_RATE_22KHZ:
            case MICROPHONE_SAMPLE_RATE_44KHZ:
            case MICROPHONE_SAMPLE_RATE_48KHZ:
                break;
            default:
                return false;
        }

        return true;
    }

    /**
     * @brief Setup I2S channel configuration
     */
    void _setupChannelConfig()
    {
        _chan_config = I2S_CHANNEL_DEFAULT_CONFIG(_i2s_port, I2S_ROLE_MASTER);
        _chan_config.auto_clear = true;
    }

    /**
     * @brief Setup PDM RX configuration
     */
    void _setupPdmConfig()
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

    /**
     * @brief Cleanup I2S resources
     */
    void _cleanupI2S()
    {
        if (_rx_handle)
        {
            i2s_channel_disable(_rx_handle);
            i2s_del_channel(_rx_handle);
            _rx_handle = nullptr;
        }
        _i2s_initialized = false;
    }

    /**
     * @brief I2S DMA callback function - simplified IRAM version
     * Store raw data pointer and signal for processing in non-IRAM context
     */
    static bool IRAM_ATTR _i2sDmaCallback(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx)
    {
        SensorMicrophone *sensor = static_cast<SensorMicrophone *>(user_ctx);
        if (!sensor || !event || !event->dma_buf)
        {
            return false;
        }

        // Store raw data for processing in non-IRAM context
        sensor->_dma_data_ptr = event->dma_buf;
        sensor->_dma_data_size = event->size;
        sensor->_new_data_flag.store(true);

        // Increment callback counter for debugging
        sensor->_dma_callback_count.fetch_add(1);

        return false; // Return false to continue receiving
    }

    /**
     * @brief Process DMA data in non-IRAM context
     * rb.push <- slot with timestamp and converted data
     */
    void _processDmaData()
    {
        if (!_new_data_flag.load() || !_audio_buffer)
        {
            return;
        }

        // Get DMA data safely
        const void *dma_ptr = const_cast<const void *>(_dma_data_ptr);
        size_t dma_size = _dma_data_size;

        if (!dma_ptr || dma_size == 0)
        {
            _new_data_flag.store(false);
            return;
        }

        _data_process_count++;

        // rb.push <- slot
        int current_slot = _audio_buffer->write_index.fetch_add(1) % RING_BUFFER_SLOTS;
        auto &slot = _audio_buffer->ring_buffer[current_slot];

        // Set timestamp (safe in non-IRAM context)
        slot.timestamp = std::chrono::steady_clock::now();

        // Convert int16_t samples to float and store in slot.frames
        const int16_t *samples = static_cast<const int16_t *>(dma_ptr);
        size_t sample_count = dma_size / sizeof(int16_t);
        if (sample_count > FRAMES_PER_RB_SLOT)
            sample_count = FRAMES_PER_RB_SLOT;

        for (size_t i = 0; i < sample_count; ++i)
        {
            slot.frames[i] = static_cast<float>(samples[i]) / 32768.0f;
        }

        // Update frontend buffer with non-overlapping data
        _updateFrontendBuffer(slot.frames.data(), sample_count);

        // Debug logging every 100 callbacks
        if (_data_process_count % 100 == 0)
        {
            int write_idx = _audio_buffer->write_index.load();
            int read_idx = _audio_buffer->read_index.load();
            LOG(INFO, "DMA Debug: callbacks=%lu, processed=%lu, write_idx=%d, read_idx=%d, slot=%d, samples=%zu",
                (unsigned long)_dma_callback_count.load(), (unsigned long)_data_process_count, write_idx, read_idx,
                current_slot, sample_count);
        }

        // Clear the flag
        _new_data_flag.store(false);
    }

    /**
     * @brief Update frontend buffer with new audio data
     */
    void _updateFrontendBuffer(const float *samples, size_t count)
    {
        if (!_audio_buffer)
            return;

        std::lock_guard<std::mutex> lock(_audio_buffer->buffer_mutex);

        size_t write_pos = _audio_buffer->frontend_write_pos.load();
        size_t available_space = FRONTEND_BUFFER_SIZE - write_pos;
        size_t samples_to_copy = std::min(count, available_space);

        // Copy samples to frontend buffer
        std::memcpy(&_audio_buffer->frontend_buffer[write_pos], samples, samples_to_copy * sizeof(float));

        // Update write position
        write_pos += samples_to_copy;
        if (write_pos >= FRONTEND_BUFFER_SIZE)
        {
            write_pos = 0; // Wrap around
        }

        _audio_buffer->frontend_write_pos.store(write_pos);
    }

    /**
     * @brief Setup I2S DMA callbacks
     */
    esp_err_t _setupDmaCallbacks()
    {
        i2s_event_callbacks_t callbacks = {
            .on_recv = _i2sDmaCallback,
            .on_recv_q_ovf = nullptr,
            .on_sent = nullptr,
            .on_send_q_ovf = nullptr,
        };

        return i2s_channel_register_event_callback(_rx_handle, &callbacks, this);
    }

    /**
     * @brief Get available data count in ring buffer
     */
    size_t _getAvailableBufferSlots() const
    {
        if (!_audio_buffer)
            return 0;

        int write_idx = _audio_buffer->write_index.load();
        int read_idx = _audio_buffer->read_index.load();

        // Calculate available slots (ring buffer logic)
        int available = write_idx - read_idx;
        if (available < 0)
            available += RING_BUFFER_SLOTS;

        return static_cast<size_t>(available);
    }

    /**
     * @brief Check if circular buffer has overrun
     */
    bool _checkBufferOverrun() const
    {
        return _getAvailableBufferSlots() >= RING_BUFFER_SLOTS - 1;
    }

    /**
     * @brief Reset ring buffer state
     */
    void _resetRingBuffer()
    {
        if (_audio_buffer)
        {
            std::lock_guard<std::mutex> lock(_audio_buffer->buffer_mutex);
            _audio_buffer->write_index.store(0);
            _audio_buffer->read_index.store(0);
            _audio_buffer->tensor_read_index.store(0);
            _audio_buffer->frontend_write_pos.store(0);
            _audio_buffer->frontend_read_pos.store(0);
        }
    }

    /**
     * @brief Get overlapping frame for feature extraction - ring buffer design
     * take { memcpy ring_buffer[read_index++ % 16] -> buf_tsr; return Tensor(buf_tsr) }
     *
     * @return core::Tensor containing the overlapping frame data
     */
    core::Tensor _getOverlappingFrame()
    {
        if (!_audio_buffer)
        {
            LOG(ERROR, "Audio buffer system not initialized");
            return core::Tensor();
        }

        std::lock_guard<std::mutex> lock(_audio_buffer->buffer_mutex);

        int current_read = _audio_buffer->tensor_read_index.load();
        int write_index = _audio_buffer->write_index.load();

        // Check if we have enough data (need 20 slots for 1 second frame: 20 * 50ms = 1000ms)
        int available_slots = write_index - current_read;
        if (available_slots < 0)
            available_slots += RING_BUFFER_SLOTS;

        int required_slots = (FRAME_SIZE + FRAMES_PER_RB_SLOT - 1) / FRAMES_PER_RB_SLOT; // Ceiling division
        if (available_slots < required_slots)
        {
            // Not enough data available
            LOG(DEBUG, "Feature extraction waiting: available=%d, required=%d, write=%d, read=%d", available_slots,
                required_slots, write_index, current_read);
            return core::Tensor();
        }

        LOG(INFO, "Feature extraction: available=%d, required=%d, extracting frame", available_slots, required_slots);

        // memcpy ring_buffer[read_index++ % 16] -> buf_tsr
        size_t samples_copied = 0;
        for (int i = 0; i < required_slots && samples_copied < FRAME_SIZE; ++i)
        {
            int slot_index = (current_read + i) % RING_BUFFER_SLOTS;
            const auto &slot = _audio_buffer->ring_buffer[slot_index];

            size_t samples_to_copy = (FRAME_SIZE - samples_copied < FRAMES_PER_RB_SLOT) ? (FRAME_SIZE - samples_copied)
                                                                                        : FRAMES_PER_RB_SLOT;
            std::memcpy(&_audio_buffer->tensor_buffer[samples_copied], slot.frames.data(),
                samples_to_copy * sizeof(float));

            samples_copied += samples_to_copy;
        }

        // Update read index for next overlapping frame (50% overlap)
        int advance_slots = required_slots / 2;
        int new_read = (current_read + advance_slots) % RING_BUFFER_SLOTS;
        _audio_buffer->tensor_read_index.store(new_read);

        // return Tensor(buf_tsr)
        auto tensor_data = std::make_shared<std::byte[]>(samples_copied * sizeof(float));
        std::memcpy(tensor_data.get(), _audio_buffer->tensor_buffer.data(), samples_copied * sizeof(float));

        return core::Tensor(core::Tensor::Type::Float32, { samples_copied }, tensor_data,
            samples_copied * sizeof(float));
    }

    /**
     * @brief Get non-overlapping frontend data
     * This function provides data for real-time display/monitoring
     *
     * @param buffer Output buffer to store the data
     * @param max_samples Maximum number of samples to read
     * @return Number of samples actually read
     */
    size_t _getFrontendData(float *buffer, size_t max_samples)
    {
        if (!_audio_buffer || !buffer)
            return 0;

        std::lock_guard<std::mutex> lock(_audio_buffer->buffer_mutex);

        size_t read_pos = _audio_buffer->frontend_read_pos.load();
        size_t write_pos = _audio_buffer->frontend_write_pos.load();

        // Calculate available data
        size_t available_samples;
        if (write_pos >= read_pos)
        {
            available_samples = write_pos - read_pos;
        }
        else
        {
            available_samples = FRONTEND_BUFFER_SIZE - read_pos + write_pos;
        }

        size_t samples_to_read = std::min(available_samples, max_samples);

        if (samples_to_read == 0)
            return 0;

        // Handle wrap-around case
        if (read_pos + samples_to_read <= FRONTEND_BUFFER_SIZE)
        {
            // No wrap-around
            std::memcpy(buffer, &_audio_buffer->frontend_buffer[read_pos], samples_to_read * sizeof(float));
        }
        else
        {
            // Wrap-around case
            size_t first_part = FRONTEND_BUFFER_SIZE - read_pos;
            size_t second_part = samples_to_read - first_part;

            std::memcpy(buffer, &_audio_buffer->frontend_buffer[read_pos], first_part * sizeof(float));
            std::memcpy(buffer + first_part, &_audio_buffer->frontend_buffer[0], second_part * sizeof(float));
        }

        // Update read position
        read_pos = (read_pos + samples_to_read) % FRONTEND_BUFFER_SIZE;
        _audio_buffer->frontend_read_pos.store(read_pos);

        return samples_to_read;
    }
};

// Static member definition
SensorMicrophone *SensorMicrophone::_this = nullptr;

} // namespace porting

#endif // MICROPHONE_HPP
