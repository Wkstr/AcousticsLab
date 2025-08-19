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

namespace algorithms { namespace node {

    class SpeechCommandsNode final: public module::MNode
    {
    public:
        SpeechCommandsNode(const core::ConfigMap &configs, module::MIOS inputs, module::MIOS outputs, int priority);

        ~SpeechCommandsNode() override;

        core::Status config(const core::ConfigMap &configs) noexcept override;

        core::Status initialize() noexcept;

        core::Status validateTensors(const module::MIOS &inputs, const module::MIOS &outputs) const noexcept;

    protected:
        core::Status forward(const module::MIOS &inputs, module::MIOS &outputs) noexcept override;

        size_t getModelOutputClasses() const noexcept;

    private:
        std::shared_ptr<hal::Engine> _engine;
        std::shared_ptr<core::Model> _model;
        std::shared_ptr<core::Model::Graph> _graph;

        int _engine_id;
        int _model_id;
        int _graph_id;

        bool _initialized;
        size_t _model_output_classes;

        core::Status copyInputData(core::Tensor *input_tensor, core::Tensor *model_input_tensor) const noexcept;
    };

    std::shared_ptr<module::MNode> createSpeechCommandsNode(const core::ConfigMap &configs, module::MIOS *inputs,
        module::MIOS *outputs, int priority);

}} // namespace algorithms::node

#endif // INFERENCE_NODE_HPP
