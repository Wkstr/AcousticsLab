#pragma once
#ifndef CMD_CFGGEDAD_HPP
#define CMD_CFGGEDAD_HPP

#include "common.hpp"
#include "task_gedad.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/logger.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace v0 {

class CmdCfgGEDAD final: public api::Command
{
    static inline constexpr const char *anomaly_thresh_path = CONTEXT_PREFIX CONTEXT_VERSION "/anomaly_thresh";
    static inline constexpr const char *window_size_path = CONTEXT_PREFIX CONTEXT_VERSION "/window_size";
    static inline constexpr const char *chunk_size_path = CONTEXT_PREFIX CONTEXT_VERSION "/chunk_size";
    static inline constexpr const char *num_chunks_path = CONTEXT_PREFIX CONTEXT_VERSION "/num_chunks";
    static inline constexpr const char *euclidean_avg_min_n_chunks_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/euclidean_avg_min_n_chunks";
    static inline constexpr const char *slide_step_path = CONTEXT_PREFIX CONTEXT_VERSION "/slide_step";
    static inline constexpr const char *chunk_extraction_after_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/chunk_extraction_after";
    static inline constexpr const char *event_report_interval_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/event_report_interval";
    static inline constexpr const char *gravity_correction_alpha_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/gravity_correction_alpha";
    static inline constexpr const char *gravity_correction_beta_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/gravity_correction_beta";
    static inline constexpr const char *abnormal_output_gpio_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/abnormal_output_gpio";
    static inline constexpr const char *abnormal_output_gpio_value_path
        = CONTEXT_PREFIX CONTEXT_VERSION "/abnormal_output_gpio_value";

public:
    static inline void preInitHook()
    {
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized()) [[unlikely]]
        {
            LOG(ERROR, "Device not registered or not initialized");
            return;
        }

        {
            float anomaly_thresh = v0::shared::anomaly_threshold;
            size_t size = sizeof(anomaly_thresh);
            auto status = device->load(hal::Device::StorageType::Internal, anomaly_thresh_path, &anomaly_thresh, size);
            if (status && size == sizeof(anomaly_thresh))
            {
                v0::shared::anomaly_threshold = anomaly_thresh;
            }
        }
        {
            int window_size = v0::shared::window_size;
            size_t size = sizeof(window_size);
            auto status = device->load(hal::Device::StorageType::Internal, window_size_path, &window_size, size);
            if (status && size == sizeof(window_size))
            {
                v0::shared::window_size = window_size;
            }
        }
        {
            int chunk_size = v0::shared::chunk_size;
            size_t size = sizeof(chunk_size);
            auto status = device->load(hal::Device::StorageType::Internal, chunk_size_path, &chunk_size, size);
            if (status && size == sizeof(chunk_size))
            {
                v0::shared::chunk_size = chunk_size;
            }
        }
        {
            int num_chunks = v0::shared::num_chunks;
            size_t size = sizeof(num_chunks);
            auto status = device->load(hal::Device::StorageType::Internal, num_chunks_path, &num_chunks, size);
            if (status && size == sizeof(num_chunks))
            {
                v0::shared::num_chunks = num_chunks;
            }
        }
        {
            int euclidean_avg_min_n_chunks = v0::shared::euclidean_avg_min_n_chunks;
            size_t size = sizeof(euclidean_avg_min_n_chunks);
            auto status = device->load(hal::Device::StorageType::Internal, euclidean_avg_min_n_chunks_path,
                &euclidean_avg_min_n_chunks, size);
            if (status && size == sizeof(euclidean_avg_min_n_chunks))
            {
                v0::shared::euclidean_avg_min_n_chunks = euclidean_avg_min_n_chunks;
            }
        }
        {
            int slide_step = v0::shared::slide_step;
            size_t size = sizeof(slide_step);
            auto status = device->load(hal::Device::StorageType::Internal, slide_step_path, &slide_step, size);
            if (status && size == sizeof(slide_step))
            {
                v0::shared::slide_step = slide_step;
            }
        }
        {
            int chunk_extraction_after = v0::shared::chunk_extraction_after;
            size_t size = sizeof(chunk_extraction_after);
            auto status = device->load(hal::Device::StorageType::Internal, chunk_extraction_after_path,
                &chunk_extraction_after, size);
            if (status && size == sizeof(chunk_extraction_after))
            {
                v0::shared::chunk_extraction_after = chunk_extraction_after;
            }
        }
        {
            int event_report_interval = v0::shared::event_report_interval;
            size_t size = sizeof(event_report_interval);
            auto status = device->load(hal::Device::StorageType::Internal, event_report_interval_path,
                &event_report_interval, size);
            if (status && size == sizeof(event_report_interval))
            {
                v0::shared::event_report_interval = event_report_interval;
            }
        }
        {
            float gravity_correction_alpha = v0::shared::gravity_correction_alpha;
            size_t size = sizeof(gravity_correction_alpha);
            auto status = device->load(hal::Device::StorageType::Internal, gravity_correction_alpha_path,
                &gravity_correction_alpha, size);
            if (status && size == sizeof(gravity_correction_alpha))
            {
                v0::shared::gravity_correction_alpha = gravity_correction_alpha;
            }
        }
        {
            float gravity_correction_beta = v0::shared::gravity_correction_beta;
            size_t size = sizeof(gravity_correction_beta);
            auto status = device->load(hal::Device::StorageType::Internal, gravity_correction_beta_path,
                &gravity_correction_beta, size);
            if (status && size == sizeof(gravity_correction_beta))
            {
                v0::shared::gravity_correction_beta = gravity_correction_beta;
            }
        }
        {
            int abnormal_output_gpio = v0::shared::abnormal_output_gpio;
            size_t size = sizeof(abnormal_output_gpio);
            auto status = device->load(hal::Device::StorageType::Internal, abnormal_output_gpio_path,
                &abnormal_output_gpio, size);
            if (status && size == sizeof(abnormal_output_gpio))
            {
                v0::shared::abnormal_output_gpio = abnormal_output_gpio;
            }
        }
        {
            int abnormal_output_gpio_value = v0::shared::abnormal_output_gpio_value;
            size_t size = sizeof(abnormal_output_gpio_value);
            auto status = device->load(hal::Device::StorageType::Internal, abnormal_output_gpio_value_path,
                &abnormal_output_gpio_value, size);
            if (status && size == sizeof(abnormal_output_gpio_value))
            {
                v0::shared::abnormal_output_gpio_value = abnormal_output_gpio_value;
            }
        }
        {
            device->gpio(hal::Device::GPIOOpType::Write, v0::shared::abnormal_output_gpio,
                !v0::shared::abnormal_output_gpio_value);
        }
    }

    CmdCfgGEDAD()
        : Command("CFGGEDAD", "Configure the Gyroscope Euclidean Distance Anomaly Detection",
              core::ConfigObjectMap {
                  { "anomaly_thresh", core::ConfigObject::createFloat("anomaly_thresh", "Anomaly detection threshold",
                                          v0::shared::anomaly_threshold, 0.0f, 1.0f) },
                  { "window_size", core::ConfigObject::createInteger("window_size", "Window size for anomaly detection",
                                       v0::shared::window_size, 32, 768) },
                  { "chunk_size", core::ConfigObject::createInteger("chunk_size", "Chunk size for anomaly detection",
                                      v0::shared::chunk_size, 32, 256) },
                  { "num_chunks", core::ConfigObject::createInteger("num_chunks",
                                      "Number of chunks for anomaly detection", v0::shared::num_chunks, 10, 100) },
                  { "euclidean_avg_min_n_chunks", core::ConfigObject::createInteger("euclidean_avg_min_n_chunks",
                                                      "Euclidean distance average from minimal N chunks",
                                                      v0::shared::euclidean_avg_min_n_chunks, 1, 20) },
                  { "slide_step", core::ConfigObject::createInteger("slide_step", "Slide step of chunks",
                                      v0::shared::slide_step, 1, 64) },
                  { "chunk_extraction_after",
                      core::ConfigObject::createInteger("chunk_extraction_after", "Chunk extraction after window size",
                          v0::shared::chunk_extraction_after, 0, 512) },
                  { "event_report_interval",
                      core::ConfigObject::createInteger("event_report_interval",
                          "Event report interval in milliseconds", v0::shared::event_report_interval, -1,
                          std::numeric_limits<int>::max()) },
                  { "gravity_correction_alpha",
                      core::ConfigObject::createFloat("gravity_correction_alpha", "Gravity correction factor alpha",
                          v0::shared::gravity_correction_alpha.load(), 1.0f, 100.0f) },
                  { "gravity_correction_beta",
                      core::ConfigObject::createFloat("gravity_correction_beta", "Gravity correction factor beta",
                          v0::shared::gravity_correction_beta.load(), -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity()) },
                  { "abnormal_output_gpio",
                      core::ConfigObject::createInteger("abnormal_output_gpio",
                          "GPIO pin for abnormal output (0 to disable)", v0::shared::abnormal_output_gpio, -1, 64) },
                  { "abnormal_output_gpio_value", core::ConfigObject::createInteger("abnormal_output_gpio_value",
                                                      "GPIO value for abnormal output (1 for high, 0 for low)",
                                                      v0::shared::abnormal_output_gpio_value, 0, 1) }

              }),
          _device(hal::DeviceRegistry::getDevice())
    {
    }

    ~CmdCfgGEDAD() noexcept override = default;

    std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id) override
    {
        auto status = STATUS_OK();
        if (v0::shared::is_training.load()) [[unlikely]]
        {
            status = STATUS(EBUSY, "Training is in progress, cannot configure GEDAD");
            goto Reply;
        }

        if (auto it = args.find("@0"); it != args.end())
        {
            if (auto anomaly_thresh = std::get_if<std::string>(&it->second);
                anomaly_thresh != nullptr && !anomaly_thresh->empty())
            {
                auto &target = _args["anomaly_thresh"];
                if ((status = target.setValue(*anomaly_thresh)))
                {
                    const auto old = v0::shared::anomaly_threshold.load();
                    float anomaly_threshold = target.getValue<float>();
                    if (old != anomaly_threshold && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, anomaly_thresh_path,
                                 &anomaly_threshold, sizeof(anomaly_threshold))))
                    {
                        goto Reply;
                    }
                    v0::shared::anomaly_threshold = anomaly_threshold;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@1"); it != args.end())
        {
            if (auto window_size = std::get_if<std::string>(&it->second);
                window_size != nullptr && !window_size->empty())
            {
                auto &target = _args["window_size"];
                if ((status = target.setValue(*window_size)))
                {
                    const auto old = v0::shared::window_size;
                    int window_size = target.getValue<int>();
                    if (old != window_size && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, window_size_path, &window_size,
                                 sizeof(window_size))))
                    {
                        goto Reply;
                    }
                    v0::shared::last_trained = false;
                    v0::shared::window_size = window_size;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@2"); it != args.end())
        {
            if (auto chunk_size = std::get_if<std::string>(&it->second); chunk_size != nullptr && !chunk_size->empty())
            {
                auto &target = _args["chunk_size"];
                if ((status = target.setValue(*chunk_size)))
                {
                    const auto old = v0::shared::chunk_size;
                    int chunk_size = target.getValue<int>();
                    if (old != chunk_size && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, chunk_size_path, &chunk_size,
                                 sizeof(chunk_size))))
                    {
                        goto Reply;
                    }
                    v0::shared::chunk_size = chunk_size;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@3"); it != args.end())
        {
            if (auto num_chunks = std::get_if<std::string>(&it->second); num_chunks != nullptr && !num_chunks->empty())
            {
                auto &target = _args["num_chunks"];
                if ((status = target.setValue(*num_chunks)))
                {
                    const auto old = v0::shared::num_chunks;
                    int num_chunks = target.getValue<int>();
                    if (old != num_chunks && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, num_chunks_path, &num_chunks,
                                 sizeof(num_chunks))))
                    {
                        goto Reply;
                    }
                    v0::shared::num_chunks = num_chunks;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@4"); it != args.end())
        {
            if (auto euclidean_avg_min_n_chunks = std::get_if<std::string>(&it->second);
                euclidean_avg_min_n_chunks != nullptr && !euclidean_avg_min_n_chunks->empty())
            {
                auto &target = _args["euclidean_avg_min_n_chunks"];
                if ((status = target.setValue(*euclidean_avg_min_n_chunks)))
                {
                    const auto old = v0::shared::euclidean_avg_min_n_chunks;
                    int euclidean_avg_min_n_chunks = target.getValue<int>();
                    if (old != euclidean_avg_min_n_chunks && _device
                        && !(
                            status = _device->store(hal::Device::StorageType::Internal, euclidean_avg_min_n_chunks_path,
                                &euclidean_avg_min_n_chunks, sizeof(euclidean_avg_min_n_chunks))))
                    {
                        goto Reply;
                    }
                    v0::shared::euclidean_avg_min_n_chunks = euclidean_avg_min_n_chunks;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@5"); it != args.end())
        {
            if (auto slide_step = std::get_if<std::string>(&it->second); slide_step != nullptr && !slide_step->empty())
            {
                auto &target = _args["slide_step"];
                if ((status = target.setValue(*slide_step)))
                {
                    const auto old = v0::shared::slide_step;
                    int slide_step = target.getValue<int>();
                    if (old != slide_step && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, slide_step_path, &slide_step,
                                 sizeof(slide_step))))
                    {
                        goto Reply;
                    }
                    v0::shared::slide_step = slide_step;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@6"); it != args.end())
        {
            if (auto chunk_extraction_after = std::get_if<std::string>(&it->second);
                chunk_extraction_after != nullptr && !chunk_extraction_after->empty())
            {
                auto &target = _args["chunk_extraction_after"];
                if ((status = target.setValue(*chunk_extraction_after)))
                {
                    const auto old = v0::shared::chunk_extraction_after;
                    int chunk_extraction_after = target.getValue<int>();
                    if (old != chunk_extraction_after && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, chunk_extraction_after_path,
                                 &chunk_extraction_after, sizeof(chunk_extraction_after))))
                    {
                        goto Reply;
                    }
                    v0::shared::chunk_extraction_after = chunk_extraction_after;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@7"); it != args.end())
        {
            if (auto event_report_interval = std::get_if<std::string>(&it->second);
                event_report_interval != nullptr && !event_report_interval->empty())
            {
                auto &target = _args["event_report_interval"];
                if ((status = target.setValue(*event_report_interval)))
                {
                    const auto old = v0::shared::event_report_interval.load();
                    int event_report_interval = target.getValue<int>();
                    if (old != event_report_interval && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, event_report_interval_path,
                                 &event_report_interval, sizeof(event_report_interval))))
                    {
                        goto Reply;
                    }
                    v0::shared::event_report_interval = event_report_interval;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@8"); it != args.end())
        {
            if (auto gravity_correction_alpha = std::get_if<std::string>(&it->second);
                gravity_correction_alpha != nullptr && !gravity_correction_alpha->empty())
            {
                auto &target = _args["gravity_correction_alpha"];
                if ((status = target.setValue(*gravity_correction_alpha)))
                {
                    const auto old = v0::shared::gravity_correction_alpha.load();
                    float gravity_correction_alpha = target.getValue<float>();
                    if (old != gravity_correction_alpha && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, gravity_correction_alpha_path,
                                 &gravity_correction_alpha, sizeof(gravity_correction_alpha))))
                    {
                        goto Reply;
                    }
                    v0::shared::gravity_correction_alpha = gravity_correction_alpha;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@9"); it != args.end())
        {
            if (auto gravity_correction_beta = std::get_if<std::string>(&it->second);
                gravity_correction_beta != nullptr && !gravity_correction_beta->empty())
            {
                auto &target = _args["gravity_correction_beta"];
                if ((status = target.setValue(*gravity_correction_beta)))
                {
                    const auto old = v0::shared::gravity_correction_beta.load();
                    float gravity_correction_beta = target.getValue<float>();
                    if (old != gravity_correction_beta && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, gravity_correction_beta_path,
                                 &gravity_correction_beta, sizeof(gravity_correction_beta))))
                    {
                        goto Reply;
                    }
                    v0::shared::gravity_correction_beta = gravity_correction_beta;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@10"); it != args.end())
        {
            if (auto abnormal_output_gpio = std::get_if<std::string>(&it->second);
                abnormal_output_gpio != nullptr && !abnormal_output_gpio->empty())
            {
                auto &target = _args["abnormal_output_gpio"];
                if ((status = target.setValue(*abnormal_output_gpio)))
                {
                    const auto old = v0::shared::abnormal_output_gpio;
                    int abnormal_output_gpio = target.getValue<int>();
                    if (old != abnormal_output_gpio && _device
                        && !(status = _device->store(hal::Device::StorageType::Internal, abnormal_output_gpio_path,
                                 &abnormal_output_gpio, sizeof(abnormal_output_gpio))))
                    {
                        goto Reply;
                    }
                    v0::shared::abnormal_output_gpio = abnormal_output_gpio;
                }
                else
                    goto Reply;
            }
        }
        if (auto it = args.find("@11"); it != args.end())
        {
            if (auto abnormal_output_gpio_value = std::get_if<std::string>(&it->second);
                abnormal_output_gpio_value != nullptr && !abnormal_output_gpio_value->empty())
            {
                auto &target = _args["abnormal_output_gpio_value"];
                if ((status = target.setValue(*abnormal_output_gpio_value)))
                {
                    const auto old = v0::shared::abnormal_output_gpio_value;
                    int abnormal_output_gpio_value = target.getValue<int>();
                    if (old != abnormal_output_gpio_value && _device
                        && !(
                            status = _device->store(hal::Device::StorageType::Internal, abnormal_output_gpio_value_path,
                                &abnormal_output_gpio_value, sizeof(abnormal_output_gpio_value))))
                    {
                        goto Reply;
                    }
                    v0::shared::abnormal_output_gpio_value = abnormal_output_gpio_value;
                }
                else
                    goto Reply;
            }
        }

        {
            _device->gpio(hal::Device::GPIOOpType::Write, v0::shared::abnormal_output_gpio,
                !v0::shared::abnormal_output_gpio_value);
        }

    Reply: {
        auto sensor = hal::SensorRegistry::getSensor(v0::defaults::accelerometer_id);
        if (sensor) [[likely]]
        {
            const auto &configs = sensor->info().configs;
            const auto it = configs.find("sr");
            if (it != configs.end())
            {
                auto val = it->second.getValue<int>();
                if (val > 0)
                    v0::shared::sample_interval_ms = static_cast<int>(1000 / val);
            }
        }
    }
        v0::shared::estimated_sample_time_ms = v0::shared::sample_interval_ms * v0::shared::window_size;

        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(transport)));

            writer["type"] += v0::ResponseType::Direct;
            writer["name"] += _name;
            if (auto it = args.find("@cmd_tag"); it != args.end())
            {
                auto tag = std::get_if<std::string>(&it->second);
                if (tag != nullptr)
                    writer["tag"] << *tag;
            }
            writer["code"] += status.code();
            writer["msg"] << status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                {
                    auto args = data["args"].writer<core::ArrayWriter>();
                    args += v0::shared::anomaly_threshold.load();
                    args << v0::shared::window_size << v0::shared::chunk_size << v0::shared::num_chunks
                         << v0::shared::euclidean_avg_min_n_chunks << v0::shared::slide_step
                         << v0::shared::chunk_extraction_after << v0::shared::event_report_interval.load();
                    args += v0::shared::gravity_correction_alpha.load();
                    args += v0::shared::gravity_correction_beta.load();
                    args += v0::shared::abnormal_output_gpio;
                    args += v0::shared::abnormal_output_gpio_value;
                }
                data["sampleInterval"] += v0::shared::sample_interval_ms;
                data["estimatedSampleTime"] += v0::shared::estimated_sample_time_ms;
            }
        }

        return nullptr;
    }

private:
    hal::Device *_device;
};

} // namespace v0

#endif // CMD_CFGGEDAD_HPP
