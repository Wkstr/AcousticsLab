#pragma once
#ifndef SPEECH_COMMANDS_NODE_HPP
#define SPEECH_COMMANDS_NODE_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "hal/engine.hpp"
#include "module/module_node.hpp"

#include <memory>
#include <string>

namespace algorithm::node {

class SpeechCommands final: public module::MNode
{
public:
    static inline constexpr const std::string_view node_name = "SpeechCommands";

    static std::shared_ptr<module::MNode> create(const core::ConfigMap &configs, const module::MIOS *inputs,
        const module::MIOS *outputs, int priority)
    {
        if (inputs && getInputMIOS(inputs).empty())
        {
            LOG(ERROR, "Failed to get or create input MIOS");
            return {};
        }

        if (outputs && getOutputMIOS(outputs).empty())
        {
            LOG(ERROR, "Failed to get or create output MIOS");
            return {};
        }

        hal::Engine *engine = hal::EngineRegistry::getEngine(1);
        if (!engine)
        {
            LOG(ERROR, "Engine not found");
            return {};
        }

        if (!engine->initialized())
        {
            auto status = engine->init();
            if (!status)
            {
                LOG(ERROR, "Engine initialization failed: %s", status.message().c_str());
                return {};
            }
        }

        auto model_info = engine->modelInfo([](const core::Model::Info &info) { return info.id == 1; });
        if (!model_info)
        {
            LOG(ERROR, "Model not found");
            return {};
        }

        std::shared_ptr<core::Model> model;
        auto status = engine->loadModel(model_info, model);
        if (!status)
        {
            LOG(ERROR, "Model loading failed: %s", status.message().c_str());
            return {};
        }

        auto graph = model->graph(0);
        if (!graph)
        {
            LOG(ERROR, "Graph not found");
            return {};
        }

        auto model_input_tensor = graph->input(0);
        auto model_output_tensor = graph->output(0);
        if (!model_input_tensor || !model_output_tensor)
        {
            LOG(ERROR, "Model graph has no input or output tensor");
            return {};
        }

        size_t model_output_classes = static_cast<size_t>(model_output_tensor->shape().dot());

        auto input_mios = getInputMIOS(inputs, model_input_tensor);
        if (input_mios.empty())
        {
            LOG(ERROR, "Failed to create or validate input tensors with model");
            return {};
        }

        auto output_mios = getOutputMIOS(outputs, model_output_classes);
        if (output_mios.empty())
        {
            LOG(ERROR, "Failed to create or validate output tensors with model");
            return {};
        }

        std::shared_ptr<module::MNode> node(new SpeechCommands(configs, std::move(input_mios), std::move(output_mios),
            priority, model, graph, model_output_classes));
        return node;
    }
    inline ~SpeechCommands() noexcept override
    {
        if (_graph)
        {
            _graph.reset();
        }
        if (_model)
        {
            _model.reset();
        }
    }

    SpeechCommands(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority,
        std::shared_ptr<core::Model> model, std::shared_ptr<core::Model::Graph> graph,
        size_t model_output_classes) noexcept
        : module::MNode(std::string(node_name), std::move(inputs), std::move(outputs), priority), _model(model),
          _graph(graph), _model_output_classes(model_output_classes)
    {
    }

private:
    static module::MIOS getInputMIOS(const module::MIOS *inputs, core::Tensor *model_input_tensor = nullptr) noexcept
    {
        if (inputs)
        {
            if (inputs->size() != 1)
                return {};
            const auto &mio = (*inputs)[0];
            if (!mio)
                return {};
            auto *tensor = mio->operator()();
            if (!tensor || !tensor->data() || tensor->dtype() != core::Tensor::Type::Float32)
                return {};
            if (model_input_tensor
                && ((tensor->shape().size() != 4 || tensor->shape()[0] != 1 || tensor->shape()[1] != 43
                     || tensor->shape()[2] != 232 || tensor->shape()[3] != 1)))
                return {};
            return *inputs;
        }

        if (model_input_tensor)
        {
            auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
                core::Tensor::Shape(1, 43, 232, 1));
            if (!tensor)
                return {};
            auto mio = std::make_shared<module::MIO>(tensor, "feature_input");
            if (!mio)
                return {};
            return { mio };
        }
        return {};
    }

    static module::MIOS getOutputMIOS(const module::MIOS *outputs, size_t model_output_classes = 0) noexcept
    {
        if (outputs)
        {
            if (outputs->size() != 1)
                return {};
            const auto &mio = (*outputs)[0];
            if (!mio)
                return {};
            auto *tensor = mio->operator()();
            if (!tensor || tensor->dtype() != core::Tensor::Type::Class)
                return {};
            if (model_output_classes > 0 && static_cast<size_t>(tensor->shape().dot()) != model_output_classes)
                return {};
            return *outputs;
        }

        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Class,
            core::Tensor::Shape(static_cast<int>(model_output_classes)));
        if (!tensor)
            return {};
        auto mio = std::make_shared<module::MIO>(tensor, "classification_output");
        if (!mio)
            return {};
        return { mio };
    }

    inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
    {
        const auto &input_tensor = inputs[0]->operator()();
        const auto &output_tensor = outputs[0]->operator()();

        auto model_input_tensor = _graph->input(0);
        auto model_output_tensor = _graph->output(0);

        auto status = copyInputData(input_tensor, model_input_tensor);
        status = _graph->forward();

        core::class_t *class_data = output_tensor->data<core::class_t>();
        if (model_output_tensor->dtype() == core::Tensor::Type::Float32)
        {
            const float *model_data = model_output_tensor->data<float>();
            for (size_t i = 0; i < _model_output_classes; ++i)
            {
                class_data[i] = { static_cast<int>(i), model_data[i] };
            }
        }
        else if (model_output_tensor->dtype() == core::Tensor::Type::Int8)
        {
            const int8_t *model_data = model_output_tensor->data<int8_t>();
            auto quant_params = _graph->outputQuantParams(0);
            float scale = quant_params.scale();
            int32_t zero_point = quant_params.zeroPoint();

            for (size_t i = 0; i < _model_output_classes; ++i)
            {
                float confidence = static_cast<float>(model_data[i] - zero_point) * scale;
                class_data[i] = { static_cast<int>(i), confidence };
            }
        }
        else
        {
            return STATUS(ENOTSUP, "Unsupported model output tensor data type");
        }

        return STATUS_OK();
    }
    inline core::Status copyInputData(core::Tensor *input_tensor, core::Tensor *model_input_tensor) const noexcept
    {
        const size_t elements_to_process = std::min(static_cast<size_t>(input_tensor->shape().dot()),
            static_cast<size_t>(model_input_tensor->shape().dot()));

        if (model_input_tensor->dtype() == core::Tensor::Type::Float32)
        {
            float *model_data = model_input_tensor->data<float>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model input data pointer");
            }

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *input_data = input_tensor->data<float>();
                if (!input_data)
                {
                    return STATUS(EFAULT, "Failed to get input data pointer");
                }
                std::memcpy(model_data, input_data, elements_to_process * sizeof(float));
            }
        }
        else if (model_input_tensor->dtype() == core::Tensor::Type::Int8)
        {
            int8_t *model_data = model_input_tensor->data<int8_t>();
            if (!model_data)
            {
                return STATUS(EFAULT, "Failed to get model input data pointer");
            }

            auto input_quant_params = _graph->inputQuantParams(0);
            float scale = input_quant_params.scale();
            int32_t zero_point = input_quant_params.zeroPoint();

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *input_data = input_tensor->data<float>();
                const float inv_scale = 1.0f / scale;
                for (size_t i = 0; i < elements_to_process; ++i)
                {
                    int32_t quantized_value = static_cast<int32_t>((input_data[i] * inv_scale)) + zero_point;
                    model_data[i] = static_cast<int8_t>(quantized_value);
                }
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported input tensor data type for int8 model");
            }
        }
        else
        {
            return STATUS(ENOTSUP, "Unsupported model input tensor data type");
        }
        return STATUS_OK();
    }

private:
    std::shared_ptr<core::Model> _model;
    std::shared_ptr<core::Model::Graph> _graph;
    size_t _model_output_classes;
};

} // namespace algorithm::node

#endif // SPEECH_COMMANDS_NODE_HPP
