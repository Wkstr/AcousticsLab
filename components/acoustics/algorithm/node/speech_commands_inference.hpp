#pragma once
#ifndef SPEECH_COMMANDS_inference_HPP
#define SPEECH_COMMANDS_inference_HPP

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
        if (inputs && !preliminaryValidateInputMIOS(*inputs))
        {
            LOG(ERROR, "User-provided input MIOS validation failed: shape or type mismatch.");
            return {};
        }

        if (outputs && !preliminaryValidateOutputMIOS(*outputs))
        {
            LOG(ERROR, "User-provided output MIOS validation failed: must be a 1D Class tensor.");
            return {};
        }

        hal::Engine *engine = hal::EngineRegistry::getEngine(1);
        if (!engine)
        {
            return {};
        }

        if (!engine->initialized())
        {
            auto status = engine->init();
            if (!status)
            {
                return {};
            }
        }

        auto model_info = engine->modelInfo([](const core::Model::Info &info) { return info.id == 1; });
        if (!model_info)
        {
            return {};
        }

        std::shared_ptr<core::Model> model;
        auto status = engine->loadModel(model_info, model);
        if (!status)
        {
            return {};
        }

        auto graph = model->graph(0);
        if (!graph)
        {
            return {};
        }

        auto model_input_tensor = graph->input(0);
        auto model_output_tensor = graph->output(0);
        if (!model_input_tensor || !model_output_tensor)
        {
            return {};
        }

        if (!postModelValidateInputTensor(model_input_tensor))
        {
            return {};
        }
        if (!postModelValidateOutputTensor(model_output_tensor))
        {
            return {};
        }

        size_t model_output_classes = static_cast<size_t>(model_output_tensor->shape().dot());
        if (outputs && !validateUserOutputWithModel(*outputs, model_output_classes))
        {
            return {};
        }

        auto input_mios = inputs ? *inputs : createInputMIOS(model_input_tensor);
        if (input_mios.empty())
        {
            return {};
        }

        auto output_mios = outputs ? *outputs : createOutputMIOS(model_output_classes);
        if (output_mios.empty())
        {
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

private:
    static inline const core::Tensor::Shape kExpectedInputShape = { 1, 43, 232, 1 };

    static bool preliminaryValidateInputMIOS(const module::MIOS &inputs) noexcept
    {
        if (inputs.size() != 1)
            return false;
        const auto &mio = inputs[0];
        if (!mio)
            return false;
        auto *tensor = mio->operator()();
        if (!tensor || !tensor->data() || tensor->dtype() != core::Tensor::Type::Float32)
            return false;
        return tensor->shape() == kExpectedInputShape;
    }

    static bool postModelValidateInputTensor(core::Tensor *model_input_tensor) noexcept
    {
        if (model_input_tensor->shape() != kExpectedInputShape)
        {
            LOG(ERROR, "Loaded model's input shape mismatch. Expected [1, 43, 232, 1].");
            return false;
        }
        return true;
    }

    static bool preliminaryValidateOutputMIOS(const module::MIOS &outputs) noexcept
    {
        if (outputs.size() != 1)
            return false;
        const auto &mio = outputs[0];
        if (!mio)
            return false;
        auto *tensor = mio->operator()();
        if (!tensor || tensor->dtype() != core::Tensor::Type::Class)
            return false;
        return tensor->shape().size() == 1;
    }

    static bool postModelValidateOutputTensor(core::Tensor *model_output_tensor) noexcept
    {
        const auto &shape = model_output_tensor->shape();
        if (shape.size() != 2 || shape[0] != 1)
        {
            return false;
        }
        return true;
    }

    static bool validateUserOutputWithModel(const module::MIOS &outputs, size_t model_output_classes) noexcept
    {
        auto *tensor = outputs[0]->operator()();
        if (static_cast<size_t>(tensor->shape().dot()) != model_output_classes)
        {
            return false;
        }
        return true;
    }

    static module::MIOS createInputMIOS(core::Tensor *model_input_tensor) noexcept
    {
        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
            model_input_tensor->shape());
        if (!tensor)
            return {};
        auto mio = std::make_shared<module::MIO>(tensor, "feature_input");
        if (!mio)
            return {};
        return { mio };
    }

    static module::MIOS createOutputMIOS(size_t model_output_classes) noexcept
    {
        auto tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Class,
            core::Tensor::Shape(static_cast<int>(model_output_classes)));
        if (!tensor)
            return {};
        auto mio = std::make_shared<module::MIO>(tensor, "classification_output");
        if (!mio)
            return {};
        return { mio };
    }

    SpeechCommands(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority,
        std::shared_ptr<core::Model> model, std::shared_ptr<core::Model::Graph> graph,
        size_t model_output_classes) noexcept
        : module::MNode(std::string(node_name), std::move(inputs), std::move(outputs), priority), _model(model),
          _graph(graph), _model_output_classes(model_output_classes)
    {
    }

    inline core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override
    {
        const auto &input_tensor = inputs[0]->operator()();
        const auto &output_tensor = outputs[0]->operator()();

        auto model_input_tensor = _graph->input(0);
        auto model_output_tensor = _graph->output(0);

        auto status = copyInputData(input_tensor, model_input_tensor);
        if (!status)
            return status;

        status = _graph->forward();
        if (!status)
            return status;

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
        const size_t elements_to_process = static_cast<size_t>(input_tensor->shape().dot());

        if (model_input_tensor->dtype() == core::Tensor::Type::Float32)
        {
            float *model_data = model_input_tensor->data<float>();
            if (!model_data)
                return STATUS(EFAULT, "Failed to get model input data pointer");

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                const float *input_data = input_tensor->data<float>();
                if (!input_data)
                    return STATUS(EFAULT, "Failed to get input data pointer");
                std::memcpy(model_data, input_data, elements_to_process * sizeof(float));
            }
            else
            {
                return STATUS(ENOTSUP, "Unsupported input tensor data type for float32 model");
            }
        }
        else if (model_input_tensor->dtype() == core::Tensor::Type::Int8)
        {
            int8_t *model_data = model_input_tensor->data<int8_t>();
            if (!model_data)
                return STATUS(EFAULT, "Failed to get model input data pointer");

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

#endif // SPEECH_COMMANDS_inference_HPP