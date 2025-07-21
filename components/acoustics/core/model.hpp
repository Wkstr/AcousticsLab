#pragma once
#ifndef MODEL_HPP
#define MODEL_HPP

#include "logger.hpp"
#include "status.hpp"
#include "tensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace core {

class Model final
{
public:
    enum class Type {
        Unknown = 0,
        Custom,
        ONNX,
        TFLite,
        TorchScript,
        TensorRT,
        HEF,
    };

    struct Info final
    {
        Info(int id, std::string name, Type type, std::string version, std::unordered_map<int, std::string> &&labels,
            std::variant<std::string, const void *> location) noexcept
            : id(id), name(std::move(name)), type(type), version(std::move(version)), labels(std::move(labels)),
              location(std::move(location))
        {
        }

        ~Info() = default;

        const int id;
        const std::string name;
        const Type type;
        const std::string version;
        std::unordered_map<int, std::string> labels;
        const std::variant<std::string, const void *> location;
    };

    struct Reporter final
    {
        Reporter() = default;
        ~Reporter() = default;

        std::unordered_map<std::string, int> perf;
    };

    class Graph final
    {
    public:
        using ForwardCallback = std::function<core::Status(Graph &)>;

        template<typename T = std::shared_ptr<Graph>, typename... Args>
        [[nodiscard]] static T create(int id, std::string name, std::vector<std::unique_ptr<Tensor>> &&inputs,
            std::vector<std::unique_ptr<Tensor>> &&outputs, std::vector<Tensor::QuantParams> &&input_quant_params,
            std::vector<Tensor::QuantParams> &&output_quant_params, ForwardCallback &&forward_callback) noexcept
        {
            if (id < 0) [[unlikely]]
            {
                LOG(ERROR, "Graph ID is negative: %d", id);
                return {};
            }
            if (name.empty()) [[unlikely]]
            {
                name = std::string("Graph_") + std::to_string(id);
                LOG(WARNING, "Graph name is empty, using default name: '%s'", name.c_str());
            }
            if (inputs.empty()) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no inputs", name.c_str());
                return {};
            }
            if (outputs.empty()) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no outputs", name.c_str());
                return {};
            }
            if (input_quant_params.size() != inputs.size()) [[unlikely]]
            {
                LOG(WARNING, "Graph '%s' input quantization parameters size mismatch: %zu != %zu", name.c_str(),
                    input_quant_params.size(), inputs.size());
            }
            if (output_quant_params.size() != outputs.size()) [[unlikely]]
            {
                LOG(WARNING, "Graph '%s' output quantization parameters size mismatch: %zu != %zu", name.c_str(),
                    output_quant_params.size(), outputs.size());
            }
            if (!forward_callback) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no forward callback", name.c_str());
                return {};
            }
            return T { new Graph(id, std::move(name), std::move(inputs), std::move(outputs),
                std::move(input_quant_params), std::move(output_quant_params), std::move(forward_callback)) };
        }

        Graph(const Graph &other) = delete;
        Graph(Graph &&other) = delete;

        Graph &operator=(const Graph &other) = delete;
        Graph &operator=(Graph &&other) = delete;

        ~Graph() = default;

        inline size_t id() const noexcept
        {
            return _id;
        }

        const std::string &name() const noexcept
        {
            return _name;
        }

        inline size_t inputs() const noexcept
        {
            return _inputs.size();
        }

        inline size_t outputs() const noexcept
        {
            return _outputs.size();
        }

        inline Tensor *input(size_t index) noexcept
        {
            if (index >= _inputs.size()) [[unlikely]]
            {
                LOG(ERROR, "Input index out of bounds: %zu >= %zu", index, _inputs.size());
                return nullptr;
            }
            return _inputs[index].get();
        }

        inline Tensor *output(size_t index) noexcept
        {
            if (index >= _outputs.size()) [[unlikely]]
            {
                LOG(ERROR, "Output index out of bounds: %zu >= %zu", index, _outputs.size());
                return nullptr;
            }
            return _outputs[index].get();
        }

        inline Tensor::QuantParams inputQuantParams(size_t index) const noexcept
        {
            if (index >= _input_quant_params.size()) [[unlikely]]
            {
                LOG(ERROR, "Input quantization parameters index out of bounds: %zu >= %zu", index,
                    _input_quant_params.size());
                return Tensor::QuantParams { 1.0f, 0 };
            }
            return _input_quant_params[index];
        }

        inline Tensor::QuantParams outputQuantParams(size_t index) const noexcept
        {
            if (index >= _output_quant_params.size()) [[unlikely]]
            {
                LOG(ERROR, "Output quantization parameters index out of bounds: %zu >= %zu", index,
                    _output_quant_params.size());
                return Tensor::QuantParams { 1.0f, 0 };
            }
            return _output_quant_params[index];
        }

        inline core::Status forward() noexcept
        {
            return _forward_callback(*this);
        }

        inline core::Status forward(Reporter &reporter) noexcept
        {
            const auto start = std::chrono::steady_clock::now();
            auto status = forward();
            const auto end = std::chrono::steady_clock::now();
            std::string key = "Graph_" + _name + "_Forward_Ms";
            reporter.perf[key] += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            return status;
        }

    private:
        Graph(int id, std::string &&name, std::vector<std::unique_ptr<Tensor>> &&inputs,
            std::vector<std::unique_ptr<Tensor>> &&outputs, std::vector<Tensor::QuantParams> &&input_quant_params,
            std::vector<Tensor::QuantParams> &&output_quant_params, ForwardCallback &&forward_callback) noexcept
            : _id(id), _name(std::move(name)), _inputs(std::move(inputs)), _outputs(std::move(outputs)),
              _input_quant_params(std::move(input_quant_params)), _output_quant_params(std::move(output_quant_params)),
              _forward_callback(std::move(forward_callback))
        {
        }

        const int _id;
        const std::string _name;
        std::vector<std::unique_ptr<Tensor>> _inputs;
        std::vector<std::unique_ptr<Tensor>> _outputs;
        std::vector<Tensor::QuantParams> _input_quant_params;
        std::vector<Tensor::QuantParams> _output_quant_params;

        ForwardCallback _forward_callback;
    };

    template<typename T>
    Model(T &&info, std::vector<std::shared_ptr<Graph>> &&graphs) noexcept
        : _info(std::forward<T>(info)), _graphs(std::move(graphs))
    {
    }

    template<typename T>
    Model(T &&info, std::shared_ptr<Graph> &&graph) noexcept
        : _info(std::forward<T>(info)), _graphs({ std::move(graph) })
    {
    }

    Model(const Model &other) = delete;
    Model(Model &&other) = delete;

    Model &operator=(const Model &other) = delete;
    Model &operator=(Model &&other) = delete;

    ~Model() noexcept
    {
        LOG(INFO, "Destroying model '%s'", _info->name.c_str());
        _graphs.clear();
        _info.reset();
    }

    const std::shared_ptr<Info> &info() const noexcept
    {
        return _info;
    }

    const std::vector<std::shared_ptr<Graph>> &graphs() const noexcept
    {
        return _graphs;
    }

    inline std::shared_ptr<Graph> graph(size_t index) noexcept
    {
        if (index >= _graphs.size()) [[unlikely]]
        {
            LOG(ERROR, "Graph index out of bounds: %zu >= %zu", index, _graphs.size());
            return nullptr;
        }
        return _graphs[index];
    }

private:
    std::shared_ptr<Info> _info;
    std::vector<std::shared_ptr<Graph>> _graphs;
};

} // namespace core

#endif // MODEL_HPP
