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

namespace algorithm { namespace node {

    class SpeechCommands final: public module::MNode
    {
    public:
        static std::shared_ptr<module::MNode> create(const core::ConfigMap &configs, const module::MIOS *inputs,
            const module::MIOS *outputs, int priority)
        {
            module::MIOS input_mios = inputs ? *inputs : module::MIOS {};
            module::MIOS output_mios = outputs ? *outputs : module::MIOS {};

            auto validate_status = validateTensors(input_mios, output_mios);
            if (!validate_status)
            {
                return nullptr;
            }

            auto node = std::shared_ptr<SpeechCommands>(
                new SpeechCommands(configs, std::move(input_mios), std::move(output_mios), priority));

            if (!node)
            {
                return nullptr;
            }

            auto init_status = node->initialize();
            if (!init_status)
            {
                return nullptr;
            }

            return node;
        }

        static core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) noexcept
        {
            if (inputs.empty() || outputs.empty())
            {
                return STATUS_OK();
            }

            if (inputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommands requires exactly 1 input tensor");
            }

            if (outputs.size() != 1)
            {
                return STATUS(EINVAL, "SpeechCommands requires exactly 1 output tensor");
            }

            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            if (!input_tensor || !output_tensor)
            {
                return STATUS(EINVAL, "Input or output tensor is null");
            }

            if (input_tensor->dtype() != core::Tensor::Type::Float32)
            {
                return STATUS(EINVAL, "Input tensor data type mismatch");
            }

            if (output_tensor->dtype() != core::Tensor::Type::Class)
            {
                return STATUS(EINVAL, "Output tensor data type mismatch");
            }

            if (input_tensor->shape().dot() <= 0)
            {
                return STATUS(EINVAL, "Input tensor has invalid shape");
            }

            if (output_tensor->shape().dot() <= 0)
            {
                return STATUS(EINVAL, "Output tensor has invalid shape");
            }

            return STATUS_OK();
        }

    private:
        inline SpeechCommands(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority)
            : module::MNode("SpeechCommands", std::move(inputs), std::move(outputs), priority), _engine(nullptr),
              _model(nullptr), _graph(nullptr), _engine_id(1), _model_id(1), _graph_id(0), _model_output_classes(0)
        {
        }

    public:
        inline ~SpeechCommands() override { }

    private:
        inline core::Status initialize() noexcept
        {
            _engine = std::shared_ptr<hal::Engine>(hal::EngineRegistry::getEngine(_engine_id), [](hal::Engine *) { });
            if (!_engine)
            {
                return STATUS(ENODEV, "Engine not found");
            }

            if (!_engine->initialized())
            {
                auto status = _engine->init();
                if (!status)
                {
                    return status;
                }
            }

            auto model_info
                = _engine->modelInfo([this](const core::Model::Info &info) { return info.id == _model_id; });
            if (!model_info)
            {
                return STATUS(ENOENT, "Model not found");
            }

            auto status = _engine->loadModel(model_info, _model);
            if (!status)
            {
                return status;
            }

            _graph = _model->graph(_graph_id);
            if (!_graph)
            {
                return STATUS(ENOENT, "Graph not found");
            }

            auto model_input_tensor = _graph->input(0);
            auto model_output_tensor = _graph->output(0);
            if (!model_input_tensor || !model_output_tensor)
            {
                return STATUS(EFAULT, "Model graph has no input or output tensor");
            }

            _model_output_classes = static_cast<size_t>(model_output_tensor->shape().dot());

            // Create default tensors if needed
            if (this->inputs().empty())
            {
                auto input_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
                    model_input_tensor->shape());
                if (!input_tensor)
                {
                    return STATUS(ENOMEM, "Failed to create input tensor");
                }
                auto input_mio = std::make_shared<module::MIO>(input_tensor, "feature_input");
                const_cast<module::MIOS &>(this->inputs()).push_back(input_mio);
            }

            if (this->outputs().empty())
            {
                auto output_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Class,
                    model_output_tensor->shape());
                if (!output_tensor)
                {
                    return STATUS(ENOMEM, "Failed to create output tensor");
                }
                auto output_mio = std::make_shared<module::MIO>(output_tensor, "classification_output");
                const_cast<module::MIOS &>(this->outputs()).push_back(output_mio);
            }

            auto tensor_validation_status = validateTensorsWithModel();
            if (!tensor_validation_status)
            {
                return tensor_validation_status;
            }

            return STATUS_OK();
        }

        inline core::Status validateTensorsWithModel() const noexcept
        {
            const auto &inputs = this->inputs();
            const auto &outputs = this->outputs();

            if (inputs.empty() || outputs.empty())
            {
                return STATUS(EINVAL, "Inputs or outputs are empty");
            }

            const auto &input_tensor = inputs[0]->operator()();
            const auto &output_tensor = outputs[0]->operator()();

            auto model_input_tensor = _graph->input(0);
            if (input_tensor->shape().dot() != model_input_tensor->shape().dot())
            {
                return STATUS(EINVAL, "Input tensor size mismatch");
            }

            auto model_output_tensor = _graph->output(0);
            size_t model_output_size = static_cast<size_t>(model_output_tensor->shape().dot());
            size_t dag_output_size = static_cast<size_t>(output_tensor->shape().dot());

            if (dag_output_size < model_output_size)
            {
                return STATUS(EINVAL, "Output tensor too small for model output");
            }
            else if (dag_output_size > model_output_size)
            {
                LOG(INFO, "Output tensor larger than model output: model=%zu, DAG=%zu (will use first %zu elements)",
                    model_output_size, dag_output_size, model_output_size);
            }
            else
            {
                LOG(INFO, "Output tensor size matches model output: %zu elements", model_output_size);
            }

            return STATUS_OK();
        }

    public:
        inline size_t getModelOutputClasses() const noexcept
        {
            return _model_output_classes;
        }

    protected:
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

    private:
        std::shared_ptr<hal::Engine> _engine;
        std::shared_ptr<core::Model> _model;
        std::shared_ptr<core::Model::Graph> _graph;

        int _engine_id;
        int _model_id;
        int _graph_id;

        size_t _model_output_classes;

        inline core::Status copyInputData(core::Tensor *input_tensor, core::Tensor *model_input_tensor) const noexcept
        {
            const size_t elements_to_process = std::min(static_cast<size_t>(input_tensor->shape().dot()),
                static_cast<size_t>(model_input_tensor->shape().dot()));

            if (elements_to_process == 0)
            {
                return STATUS(EINVAL, "No elements to copy");
            }

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
    };

}} // namespace algorithm::node

#endif // SPEECH_COMMANDS_NODE_HPP
