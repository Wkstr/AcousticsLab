#pragma once
#ifndef TASK_GEDAD_HPP
#define TASK_GEDAD_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>

namespace v0 {

namespace shared {

    inline std::atomic<bool> last_trained = false;
    inline std::atomic<bool> is_training = false;
    inline std::atomic<bool> is_sampling = false;
    inline std::atomic<bool> is_invoking = false;

    inline constexpr const size_t buffer_size = 1024;
    inline constexpr const size_t buffer_size_mask = buffer_size - 1;
    inline std::mutex buffer_mutex;
    inline size_t buffer_head = 0;
    inline size_t buffer_tail = 0;
    inline std::array<std::vector<float>, 3> buffer = {
        std::vector<float>(buffer_size),
        std::vector<float>(buffer_size),
        std::vector<float>(buffer_size),
    };
    inline std::chrono::steady_clock::time_point buffer_base_ts = std::chrono::steady_clock::now();

    inline std::atomic<float> anomaly_threshold = 0.5f;
    inline int window_size = 192;
    inline int chunk_size = 64;
    inline int num_chunks = 20;
    inline int euclidean_avg_min_n_chunks = 5;
    inline int slide_step = 1;
    inline int chunk_extraction_after = 64;
    inline std::atomic<int> event_report_interval = 100;
    inline std::atomic<float> gravity_correction_alpha = 9.78762f;
    inline std::atomic<float> gravity_correction_beta = 0.0f;

    inline int sample_interval_ms = 10;
    inline int estimated_sample_time_ms = sample_interval_ms * window_size;
    inline constexpr const size_t contiguous_n_default = 2;

    inline std::mutex gedad_mutex;
    inline std::array<std::vector<float>, 3> window;
    inline std::array<float, 3> euclidean_dist_thresh;
    inline std::array<size_t, 3> contiguous_n;

    inline int abnormal_output_gpio = -1;
    inline int abnormal_output_gpio_value = 1;

} // namespace shared

struct TaskGEDAD final
{
    class Storage final: public api::Task
    {
        static inline constexpr const char *gedad_storage_path = CONTEXT_PREFIX CONTEXT_VERSION "/gedad_storage";

    public:
        Storage(api::Context &context, hal::Transport &transport, size_t id, std::string tag) noexcept
            : api::Task(context, transport, id, v0::defaults::task_priority), _tag(std::move(tag))
        {
        }

        ~Storage() noexcept override { }

        static inline void preInitHook()
        {
            auto device = hal::DeviceRegistry::getDevice();
            if (!device || !device->initialized()) [[unlikely]]
            {
                LOG(ERROR, "Device not registered or not initialized");
                return;
            }

            const std::lock_guard<std::mutex> lock(shared::gedad_mutex);

            {
                size_t channels = 0;
                size_t size = sizeof(channels);
                auto status = device->load(hal::Device::StorageType::Internal, getChannelsPath(), &channels, size);
                if (!status || size != sizeof(channels) || channels != shared::window.size()) [[unlikely]]
                {
                    LOG(ERROR, "Failed to load GEDAD channels or size mismatch, error: %s", status.message().c_str());
                    return;
                }
            }

            {
                size_t size = shared::euclidean_dist_thresh.size() * sizeof(float);
                auto status = device->load(hal::Device::StorageType::Internal, getEuclideanDistThreshPath(),
                    shared::euclidean_dist_thresh.data(), size);
                if (!status || size != shared::euclidean_dist_thresh.size() * sizeof(float)) [[unlikely]]
                {
                    LOG(ERROR, "Failed to load GEDAD euclidean distance thresholds or size mismatch, error: %s",
                        status.message().c_str());
                    return;
                }
            }

            {
                size_t size = shared::contiguous_n.size() * sizeof(size_t);
                auto status = device->load(hal::Device::StorageType::Internal, getContiguousNPath(),
                    shared::contiguous_n.data(), size);
                if (!status || size != shared::contiguous_n.size() * sizeof(size_t)) [[unlikely]]
                {
                    LOG(ERROR, "Failed to load GEDAD contiguous_n or size mismatch, error: %s",
                        status.message().c_str());
                    return;
                }
            }

            {
                for (size_t i = 0; i < shared::window.size(); ++i)
                {
                    size_t window_size = 0;
                    size_t size = sizeof(window_size);
                    auto status = device->load(hal::Device::StorageType::Internal,
                        getWindowPath(std::to_string(i) + "_size"), &window_size, size);
                    if (!status || size != sizeof(window_size) || window_size == 0) [[unlikely]]
                    {
                        LOG(ERROR, "Failed to load GEDAD window size or size mismatch, error: %s",
                            status.message().c_str());
                        return;
                    }
                    shared::window[i].resize(window_size);
                    size = window_size * sizeof(float);
                    status = device->load(hal::Device::StorageType::Internal, getWindowPath(std::to_string(i)),
                        shared::window[i].data(), size);
                    if (!status || size != window_size * sizeof(float)) [[unlikely]]
                    {
                        LOG(ERROR, "Failed to load GEDAD window data or size mismatch, error: %s",
                            status.message().c_str());
                        return;
                    }
                }
            }

            shared::last_trained.store(true);
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            if (!shared::last_trained.load()) [[unlikely]]
            {
                return replyWithStatus(STATUS(ENODATA, "GEDAD training is not completed or canceled"));
            }
            auto device = hal::DeviceRegistry::getDevice();
            if (!device || !device->initialized()) [[unlikely]]
            {
                return replyWithStatus(STATUS(ENODEV, "Device not registered or not initialized"));
            }

            const std::lock_guard<std::mutex> lock(shared::gedad_mutex);

            {
                const std::string windows_prefix = std::string(gedad_storage_path) + "/window_";
                for (size_t i = 0; i < shared::window.size(); ++i)
                {
                    const size_t window_size = shared::window[i].size();
                    auto status = device->store(hal::Device::StorageType::Internal, getWindowPath(std::to_string(i)),
                        shared::window[i].data(), window_size * sizeof(float));
                    if (!status) [[unlikely]]
                    {
                        return replyWithStatus(status);
                    }
                    status = device->store(hal::Device::StorageType::Internal,
                        getWindowPath(std::to_string(i) + "_size"), &window_size, sizeof(window_size));
                    if (!status) [[unlikely]]
                    {
                        return replyWithStatus(status);
                    }
                }
            }

            {
                const size_t euclidean_dist_thresh_size = shared::euclidean_dist_thresh.size();
                if (auto status = device->store(hal::Device::StorageType::Internal, getEuclideanDistThreshPath(),
                        shared::euclidean_dist_thresh.data(), euclidean_dist_thresh_size * sizeof(float));
                    !status) [[unlikely]]
                {
                    return replyWithStatus(status);
                }
            }

            {
                const size_t contiguous_n_size = shared::contiguous_n.size();
                if (auto status = device->store(hal::Device::StorageType::Internal, getContiguousNPath(),
                        shared::contiguous_n.data(), contiguous_n_size * sizeof(size_t));
                    !status) [[unlikely]]
                {
                    return replyWithStatus(status);
                }
            }

            {
                const size_t channels = shared::window.size();
                if (auto status
                    = device->store(hal::Device::StorageType::Internal, getChannelsPath(), &channels, sizeof(channels));
                    !status)
                {
                    return replyWithStatus(status);
                }
            }

            return replyWithStatus(STATUS_OK());
        }

    private:
        static inline std::string getWindowPath(std::string suffix)
        {
            return std::string(gedad_storage_path) + "/window_" + suffix;
        }

        static inline std::string getEuclideanDistThreshPath()
        {
            return std::string(gedad_storage_path) + "/euclidean_dist_thresh";
        }

        static inline std::string getContiguousNPath()
        {
            return std::string(gedad_storage_path) + "/contiguous_n";
        }

        static inline std::string getChannelsPath()
        {
            return std::string(gedad_storage_path) + "/channels";
        }

        inline core::Status replyWithStatus(core::Status status)
        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v0::ResponseType::Event;
            writer["name"] += "TRAINGEDAD";
            writer["tag"] += _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();

            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["euclideanDistThresh"] += shared::euclidean_dist_thresh;
                data["contiguousN"] += shared::contiguous_n;
            }

            return status;
        }

        const std::string _tag;
    };

    class Train final: public api::Task
    {
    public:
        Train(api::Context &context, hal::Transport &transport, size_t id, std::string tag,
            size_t samples_required) noexcept
            : api::Task(context, transport, id, v0::defaults::task_priority), _tag(tag),
              _samples_required(samples_required)
        {
            shared::is_training.store(true);
            {
                const std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                v0::shared::buffer_tail = v0::shared::buffer_head;
            }
        }

        ~Train() noexcept override
        {
            shared::is_training.store(false);
        }

        core::Status operator()(api::Executor &executor) override
        {
            if (!shared::is_sampling.load()) [[unlikely]]
            {
                return replyWithStatus(STATUS(ENODATA, "Training requires sampling to be started"));
            }

            const std::lock_guard<std::mutex> train_lock(shared::gedad_mutex);

            // assign the window buffer
            {
                const std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                const size_t window_start = shared::buffer_tail;
                const size_t available = shared::buffer_head - window_start;
                if (available < _samples_required) [[unlikely]]
                {
                    return executor.submit(getptr(), getNextDataDelay(available));
                }
                for (size_t i = 0; i < shared::window.size(); ++i)
                {
                    auto &window_i = shared::window[i];
                    window_i.resize(_samples_required);
                    window_i.shrink_to_fit();
                    auto &buffer_i = shared::buffer[i];
                    for (size_t j = 0; j < _samples_required; ++j)
                    {
                        window_i[j] = buffer_i[(window_start + j) & shared::buffer_size_mask];
                    }
                }
                shared::buffer_tail += _samples_required;
            }

            // from range sample_start to sample_end, randomly select batch_size views, each view has chunk_size
            // elements
            const size_t sample_start = shared::chunk_extraction_after;
            if (sample_start > _samples_required) [[unlikely]]
            {
                return replyWithStatus(
                    STATUS(EINVAL, "Samples required must be greater than or equal to chunk extraction after"));
            }
            const size_t view_size = shared::chunk_size;
            if (view_size > _samples_required) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Chunk size must be less than or equal to samples required"));
            }
            const size_t sample_end = _samples_required - view_size;
            const size_t num_chunks = shared::num_chunks;
            if (sample_start + num_chunks > sample_end) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Not enough samples for the number of chunks"));
            }
            std::vector<size_t> view_index = generateRandomViewIndex(sample_start, sample_end, num_chunks);

            // calculate the euclidean distance for each channel
            const size_t shift_dist = shared::slide_step;
            std::array<std::vector<float>, shared::window.size()> euclidean_dist_per_channel;
            {
                const size_t window_size = shared::window_size;
                if (window_size < view_size) [[unlikely]]
                {
                    return replyWithStatus(STATUS(EINVAL, "Window size must be greater than or equal to chunk size"));
                }
                const size_t window_end = window_size - view_size;
                if (shift_dist == 0 || shift_dist > window_size) [[unlikely]]
                {
                    return replyWithStatus(
                        STATUS(EINVAL, "Slide step must be greater than 0 and less than or equal to window size"));
                }
                const size_t views_per_channel = window_size / shift_dist;
                const size_t minimal_n = shared::euclidean_avg_min_n_chunks;
                if (minimal_n > views_per_channel) [[unlikely]]
                {
                    return replyWithStatus(
                        STATUS(EINVAL, "Minimal N chunks must be less than or equal to views per channel"));
                }

                const std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                const size_t head = shared::buffer_head;
                for (size_t i = 0; i < euclidean_dist_per_channel.size(); ++i)
                {
                    // for each view, sum the total euclidean distance
                    // by sliding the view with shift_dist on the channel buffer
                    const auto &buffer_i = shared::buffer[i];
                    const auto &window_i = shared::window[i];
                    std::vector<float> euclidean_dist_per_view;
                    euclidean_dist_per_view.reserve(view_index.size());
                    for (auto j: view_index)
                    {
                        std::vector<float> euclidean_dist_per_slide;
                        euclidean_dist_per_slide.reserve(views_per_channel);

                        const auto &buffer_ij_index = head + j;
                        // slide view on the channel buffer
                        for (size_t k = 0; k < window_end; k += shift_dist)
                        {
                            // calculate the euclidean distance
                            float euclidean_dist = 0.0f;
                            for (size_t l = 0; l < view_size; ++l)
                            {
                                euclidean_dist += static_cast<float>(std::pow(
                                    static_cast<float>(
                                        buffer_i[(buffer_ij_index + l) & shared::buffer_size_mask] - window_i[k + l]),
                                    2));
                            }
                            euclidean_dist_per_slide.emplace_back(std::sqrt(euclidean_dist));
                        }
                        // sort and take minimal_n euclidean distances
                        std::sort(std::begin(euclidean_dist_per_slide), std::end(euclidean_dist_per_slide),
                            std::less<float> {});
                        std::erase_if(euclidean_dist_per_slide, [](const float &value) {
                            return std::isnan(value) || std::isinf(value)
                                   || std::abs(value) < std::numeric_limits<float>::epsilon();
                        });
                        euclidean_dist_per_slide.resize(std::min(euclidean_dist_per_slide.size(), minimal_n));
                        euclidean_dist_per_slide.shrink_to_fit();
                        // calculate the average of the minimal_n euclidean distances
                        euclidean_dist_per_view.emplace_back(
                            std::accumulate(std::begin(euclidean_dist_per_slide), std::end(euclidean_dist_per_slide),
                                std::numeric_limits<float>::epsilon())
                            / static_cast<float>(euclidean_dist_per_slide.size()));
                    }
                    euclidean_dist_per_channel[i].swap(euclidean_dist_per_view);
                }
            }

            // calculate and assign the final euclidean distance threshold
            {
                for (size_t i = 0; i < euclidean_dist_per_channel.size(); ++i)
                {
                    const auto &euclidean_dist_per_channel_i = euclidean_dist_per_channel[i];
                    const auto euclidean_dist = std::accumulate(std::begin(euclidean_dist_per_channel_i),
                                                    std::end(euclidean_dist_per_channel_i), 0.0f)
                                                / static_cast<float>(euclidean_dist_per_channel_i.size());
                    shared::euclidean_dist_thresh[i] = std::max(euclidean_dist, std::numeric_limits<float>::epsilon());
                }
            }

            // slide chunk on the window to find optimal contiguous n
            {
                for (size_t i = 0; i < shared::window.size(); ++i)
                {
                    const auto &window_i = shared::window[i];
                    const auto window_end = window_i.size() - view_size;
                    const auto &euclidean_dist_thresh_i = shared::euclidean_dist_thresh[i];
                    std::vector<size_t> contiguous_records;
                    for (auto j: view_index)
                    {
                        size_t contiguous = 0;
                        for (size_t k = 0; k < window_end; k += shift_dist)
                        {
                            float euclidean_dist = 0.0f;
                            for (size_t l = 0; l < view_size; ++l)
                            {
                                euclidean_dist += static_cast<float>(
                                    std::pow(static_cast<float>(window_i[j + l] - window_i[k + l]), 2));
                            }
                            euclidean_dist = std::sqrt(euclidean_dist);
                            if (std::isnan(euclidean_dist) || std::isinf(euclidean_dist)) [[unlikely]]
                            {
                                break;
                            }
                            if (euclidean_dist < euclidean_dist_thresh_i)
                            {
                                ++contiguous;
                            }
                            else
                            {
                                if (contiguous > 1)
                                {
                                    contiguous_records.emplace_back(contiguous);
                                }
                                contiguous = 0;
                            }
                        }
                    }

                    size_t contiguous_median = shared::contiguous_n_default;
                    if (!contiguous_records.empty()) [[likely]]
                    {
                        std::sort(std::begin(contiguous_records), std::end(contiguous_records),
                            std::greater<size_t> {});
                        contiguous_median = contiguous_records[static_cast<size_t>(contiguous_records.size() / 2)];
                    }
                    shared::contiguous_n[i] = contiguous_median;
                }
            }

            shared::last_trained.store(true);

            return replyWithStatus(STATUS_OK());
        }

    private:
        inline size_t getNextDataDelay(size_t current_available) const noexcept
        {
            if (current_available >= _samples_required) [[unlikely]]
            {
                return 0;
            }
            return v0::shared::sample_interval_ms * (_samples_required - current_available);
        }

        inline std::vector<size_t> generateRandomViewIndex(size_t range_start, size_t range_end,
            size_t n) const noexcept
        {
            std::vector<size_t> view_index(n);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dis(range_start, range_end);
            size_t rand_index;
            for (auto it = std::begin(view_index); it != std::end(view_index);)
            {
                rand_index = dis(gen);
                if (std::find(std::begin(view_index), it, rand_index) != it)
                {
                    continue; // skip if the index is already in the view_index
                }
                *it = rand_index;
                ++it;
            }

            return view_index;
        }

        inline core::Status replyWithStatus(core::Status status, bool lock_required = false)
        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v0::ResponseType::Event;
            writer["name"] += "TRAINGEDAD";
            writer["tag"] += _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();

            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                if (lock_required)
                {
                    std::lock_guard<std::mutex> lock(shared::gedad_mutex);
                    data["euclideanDistThresh"] += shared::euclidean_dist_thresh;
                    data["contiguousN"] += shared::contiguous_n;
                }
                else
                {
                    data["euclideanDistThresh"] += shared::euclidean_dist_thresh;
                    data["contiguousN"] += shared::contiguous_n;
                }
            }

            return status;
        }

        const std::string _tag;
        const size_t _samples_required;
    };

    class Invoke final: public api::Task
    {
    public:
        Invoke(api::Context &context, hal::Transport &transport, size_t id, std::string tag,
            const volatile size_t &external_task_id) noexcept
            : api::Task(context, transport, id, v0::defaults::task_priority), _tag(tag),
              _external_task_id(external_task_id), _internal_task_id(_external_task_id), _start_time(),
              _device(hal::DeviceRegistry::getDevice()), _view_size(v0::shared::chunk_size),
              _shift_dist(v0::shared::slide_step), _window_slide_step(_shift_dist), _current_id(0),
              _anomaly_score(1.0f), _perf_ms(0), _last_sync_time(std::chrono::steady_clock::now()),
              _last_report_time(_last_sync_time)
        {
            shared::is_invoking = true;
            _window_slide_step = std::min(_window_slide_step, _view_size);
            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                v0::shared::buffer_tail = v0::shared::buffer_head;
                _start_time = v0::shared::buffer_base_ts;
                _current_id = v0::shared::buffer_tail;
            }
        }

        ~Invoke() noexcept override
        {
            shared::is_invoking = false;
        }

        inline core::Status operator()(api::Executor &executor) override
        {
            shared::is_invoking = true;

            if (_external_task_id != _internal_task_id) [[unlikely]]
            {
                return replyWithStatus(STATUS_OK());
            }
            if (!shared::is_sampling.load()) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Sampling is not active"));
            }
            if (!shared::last_trained.load()) [[unlikely]]
            {
                return replyWithStatus(STATUS(ENODATA, "GEDAD training is not completed or canceled"));
            }

            const auto s = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                const auto head = shared::buffer_head;
                const auto tail = shared::buffer_tail;
                if (_current_id > head) [[unlikely]]
                {
                    _current_id = tail;
                }

                const size_t available = head - _current_id;
                if (available < _view_size) [[unlikely]]
                {
                    return executor.submit(getptr(), getNextDataDelay(available));
                }
                if (shared::buffer.size() != _cached_view.size()) [[unlikely]]
                {
                    return replyWithStatus(STATUS(EFAULT, "Cached view size mismatch"));
                }
                for (size_t i = 0; i < _cached_view.size(); ++i)
                {
                    auto &view_per_channel_i = _cached_view[i];
                    // check cache buffer size, resize if necessary
                    if (view_per_channel_i.size() != _view_size) [[unlikely]]
                    {
                        view_per_channel_i.resize(_view_size);
                        view_per_channel_i.shrink_to_fit();
                    }
                    const auto &buffer_i = shared::buffer[i];
                    for (size_t j = 0; j < _view_size; ++j)
                    {
                        view_per_channel_i[j] = buffer_i[(_current_id + j) & shared::buffer_size_mask];
                    }
                }
                if (available > (_view_size << 1))
                {
                    _current_id = shared::buffer_head - _view_size;
                }
                else
                {
                    _current_id += _window_slide_step;
                }
                shared::buffer_tail = _current_id;
            }
            _last_sync_time = s;

            const float anomaly_threshold = std::max(0.0f, std::min(shared::anomaly_threshold.load(), 1.0f));

            std::array<float, 3> anormalities = { 1.0f, 1.0f, 1.0f };
            {
                // calculate the euclidean distance for each channel
                // slide the view with shift_dist on the window buffer

                const std::lock_guard<std::mutex> invoke_lock(shared::gedad_mutex);

                if (anormalities.size() != shared::window.size()
                    || anormalities.size() != shared::euclidean_dist_thresh.size()) [[unlikely]]
                {
                    return replyWithStatus(STATUS(EFAULT, "Anormalities size mismatch"));
                }

                for (size_t i = 0; i < anormalities.size(); ++i)
                {
                    const auto &window_i = shared::window[i];
                    const auto &view_i = _cached_view[i];
                    auto &anormality_i = anormalities[i];
                    const auto thresh_i = shared::euclidean_dist_thresh[i];
                    if (window_i.size() < _view_size) [[unlikely]]
                    {
                        return replyWithStatus(STATUS(EFAULT, "Window size is less than view size"));
                    }
                    const size_t window_end = window_i.size() - _view_size;
                    int contiguous = 0;
                    for (size_t j = 0; j < window_end; j += _shift_dist)
                    {
                        float euclidean_dist = 0.0f;
                        for (size_t k = 0; k < _view_size; ++k)
                        {
                            euclidean_dist
                                += static_cast<float>(std::pow(static_cast<float>(view_i[k] - window_i[j + k]), 2));
                        }
                        euclidean_dist = std::sqrt(euclidean_dist);
                        if (std::isnan(euclidean_dist) || std::isinf(euclidean_dist))
                        {
                            break;
                        }
                        if (euclidean_dist < thresh_i)
                        {
                            if (++contiguous > shared::contiguous_n[i])
                            {
                                anormality_i = 0.0f;
                                break;
                            }
                        }
                        else
                        {
                            contiguous = 0;
                            anormality_i = std::min(anormality_i, 1.0f - (thresh_i / euclidean_dist));
                        }
                    }
                }
            }

            const float sum = std::accumulate(std::begin(anormalities), std::end(anormalities), 0.0f);
            const float anomaly_score = sum < anomaly_threshold * static_cast<float>(anormalities.size()) ? 0.0f : 1.0f;

            if (_device && shared::abnormal_output_gpio >= 0)
            {
                _device->gpio(hal::Device::GPIOOpType::Write, shared::abnormal_output_gpio,
                    anomaly_score >= anomaly_threshold ? shared::abnormal_output_gpio_value
                                                       : !shared::abnormal_output_gpio_value);
            }

            const auto e = std::chrono::steady_clock::now();
            _perf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
            {
                const auto duration
                    = std::chrono::duration_cast<std::chrono::milliseconds>(e - _last_sync_time).count();
                const size_t new_step
                    = std::ceil(static_cast<float>(duration) / static_cast<float>(v0::shared::sample_interval_ms));
                if (new_step > 0) [[likely]]
                {
                    _window_slide_step = std::min(new_step, _view_size);
                }
            }

            if (anomaly_score < 0.5f && std::abs(anomaly_score - _anomaly_score) < 0.05f)
            {
                const int report_interval = shared::event_report_interval.load();
                const auto duration
                    = std::chrono::duration_cast<std::chrono::milliseconds>(e - _last_report_time).count();
                if (duration >= report_interval) [[unlikely]]
                {
                    _anomaly_score = anomaly_score;
                    _last_report_time = e;
                    replyWithStatus(STATUS_OK());
                }
            }
            else
            {
                _anomaly_score = anomaly_score;
                _last_report_time = e;
                replyWithStatus(STATUS_OK());
            }

            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                return executor.submit(getptr(), getNextDataDelay(shared::buffer_head - shared::buffer_tail));
            }
        }

    private:
        inline core::Status replyWithStatus(core::Status status) noexcept
        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v0::ResponseType::Stream;
            writer["name"] += "ClassifyResult";
            writer["tag"] += _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();

            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["id"] += _current_id;
                data["ts"] += std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - _start_time)
                                  .count();
                {
                    auto cls_data = data["data"].writer<core::ArrayWriter>();
                    {
                        auto nomal_cls = cls_data.writer<core::ArrayWriter>();
                        nomal_cls << 0 << static_cast<float>(1.0f - _anomaly_score);
                    }
                    {
                        auto anom_cls = cls_data.writer<core::ArrayWriter>();
                        anom_cls << 1 << static_cast<float>(_anomaly_score);
                    }
                }
                data["perfMs"] += _perf_ms;
            }

            return status;
        }

        inline size_t getNextDataDelay(size_t current_available) const noexcept
        {
            if (current_available >= _view_size) [[unlikely]]
            {
                return 0;
            }
            return v0::shared::sample_interval_ms * (_view_size - current_available);
        }

        const std::string _tag;
        const volatile size_t &_external_task_id;
        const size_t _internal_task_id;
        std::chrono::steady_clock::time_point _start_time;
        hal::Device *_device;

        const size_t _view_size;
        const size_t _shift_dist;
        size_t _window_slide_step;
        size_t _current_id;
        float _anomaly_score;
        size_t _perf_ms;
        std::chrono::steady_clock::time_point _last_sync_time;
        std::chrono::steady_clock::time_point _last_report_time;
        std::array<std::vector<float>, 3> _cached_view;
    };
};

} // namespace v0

#endif // TASK_GEDAD_HPP
