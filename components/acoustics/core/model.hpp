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
        Info(Type type, std::string name, std::string version, std::unordered_map<int, std::string> &&labels,
            std::string path)
            : type(type), name(std::move(name)), version(std::move(version)), labels(std::move(labels)),
              path(std::move(path))
        {
        }

        ~Info() = default;

        const Type type;
        const std::string name;
        const std::string version;
        const std::unordered_map<int, std::string> labels;
        const std::string path;
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

        Graph(int id, std::string name, std::vector<Tensor> &&inputs, std::vector<Tensor> &&outputs,
            std::vector<Tensor::QuantParams> &&input_quant_params,
            std::vector<Tensor::QuantParams> &&output_quant_params,
            ForwardCallback &&forward_callback = nullptr) noexcept
            : _id(id), _name(std::move(name)), _inputs(std::move(inputs)), _outputs(std::move(outputs)),
              _input_quant_params(std::move(input_quant_params)), _output_quant_params(std::move(output_quant_params)),
              _forward_callback(std::move(forward_callback))
        {
            if (_id < 0) [[unlikely]]
            {
                LOG(WARNING, "Graph ID is negative: %d", _id);
            }
            if (_name.empty()) [[unlikely]]
            {
                _name = std::string("Graph_") + std::to_string(_id);
                LOG(INFO, "Graph name is empty, using default name: '%s'", _name.c_str());
            }
            if (_inputs.empty()) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no inputs", this->_name.c_str());
            }
            if (_outputs.empty()) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no outputs", this->_name.c_str());
            }
            if (_input_quant_params.size() != _inputs.size()) [[unlikely]]
            {
                LOG(WARNING, "Graph '%s' input quantization parameters size mismatch: %zu != %zu", this->_name.c_str(),
                    _input_quant_params.size(), _inputs.size());
            }
            if (_output_quant_params.size() != _outputs.size()) [[unlikely]]
            {
                LOG(WARNING, "Graph '%s' output quantization parameters size mismatch: %zu != %zu", this->_name.c_str(),
                    _output_quant_params.size(), _outputs.size());
            }
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
            return &_inputs[index];
        }

        inline Tensor *output(size_t index) noexcept
        {
            if (index >= _outputs.size()) [[unlikely]]
            {
                LOG(ERROR, "Output index out of bounds: %zu >= %zu", index, _outputs.size());
                return nullptr;
            }
            return &_outputs[index];
        }

        inline Tensor::QuantParams inputQuantParams(size_t index) const noexcept
        {
            if (index >= _input_quant_params.size()) [[unlikely]]
            {
                LOG(ERROR, "Input quantization parameters index out of bounds: %zu >= %zu", index,
                    _input_quant_params.size());
                return { 1.0f, 0 };
            }
            return _input_quant_params[index];
        }

        inline Tensor::QuantParams outputQuantParams(size_t index) const noexcept
        {
            if (index >= _output_quant_params.size()) [[unlikely]]
            {
                LOG(ERROR, "Output quantization parameters index out of bounds: %zu >= %zu", index,
                    _output_quant_params.size());
                return { 1.0f, 0 };
            }
            return _output_quant_params[index];
        }

        inline core::Status forward() noexcept
        {
            if (!_forward_callback) [[unlikely]]
            {
                LOG(ERROR, "Graph '%s' has no forward callback", _name.c_str());
                return STATUS(EFAULT, "Graph forward callback is not set");
            }
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
        const int _id;
        std::string _name;
        std::vector<Tensor> _inputs;
        std::vector<Tensor> _outputs;
        std::vector<Tensor::QuantParams> _input_quant_params;
        std::vector<Tensor::QuantParams> _output_quant_params;

        ForwardCallback _forward_callback;
    };

    Model(std::shared_ptr<Info> info, std::vector<std::shared_ptr<Graph>> &&graphs, void *data = nullptr,
        size_t size = 0, bool internal_managed = false) noexcept
        : _info(std::move(info)), _graphs(std::move(graphs)), _data(data), _size(size),
          _internal_managed(internal_managed)
    {
        if (!_info) [[unlikely]]
        {
            LOG(ERROR, "Model info is null");
        }
        if (_graphs.empty()) [[unlikely]]
        {
            LOG(ERROR, "Model '%s' has no graphs", _info->name.c_str());
        }
        if (_data && _size > 0) [[unlikely]]
        {
            LOG(ERROR, "Model '%s' has non-zero size (%zu) but no data", _info->name.c_str(), _size);
        }
    }

    Model(const Model &other) = delete;
    Model(Model &&other) = delete;

    Model &operator=(const Model &other) = delete;
    Model &operator=(Model &&other) = delete;

    ~Model() noexcept
    {
        if (_data && _internal_managed)
        {
            delete[] static_cast<std::byte *>(_data);
            _data = nullptr;
            _size = 0;
            LOG(DEBUG, "Model '%s' data memory released", _info->name.c_str());
        }
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

    inline void *data() const noexcept
    {
        return _data;
    }

    inline size_t size() const noexcept
    {
        return _size;
    }

private:
    std::shared_ptr<Info> _info;
    std::vector<std::shared_ptr<Graph>> _graphs;

    void *_data;
    size_t _size;
    const bool _internal_managed;
};

} // namespace core

#endif // MODEL_HPP
