#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "algorithms/node/speech_commands_node.hpp"
#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"
#include "module/node/node_input.hpp"
#include "module/node/node_output.hpp"

#include <memory>
#include <new>

namespace algorithms { namespace dag {

    inline std::shared_ptr<module::MDAG> createSoundClassification(const core::ConfigMap &configs)
    {
        auto dag = std::make_shared<module::MDAG>("SoundClassification");

        constexpr int AUDIO_SAMPLES = 44032;
        constexpr int FEATURE_COUNT = 9976;

        auto input_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Int16,
            core::Tensor::Shape(AUDIO_SAMPLES));
        auto feature_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Float32,
            core::Tensor::Shape(FEATURE_COUNT));

        if (!input_tensor || !feature_tensor)
        {
            LOG(ERROR, "Failed to create input or feature tensors");
            return nullptr;
        }

        auto input_mio = std::make_shared<module::MIO>(input_tensor, "audio_input");
        auto feature_mio = std::make_shared<module::MIO>(feature_tensor, "feature_data");

        constexpr int DEFAULT_OUTPUT_CLASSES = 20;
        auto output_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Class,
            core::Tensor::Shape(1, DEFAULT_OUTPUT_CLASSES));
        if (!output_tensor)
        {
            LOG(ERROR, "Failed to create output tensor");
            return nullptr;
        }
        auto output_mio = std::make_shared<module::MIO>(output_tensor, "classification_output");

        module::MIOS inference_inputs = { feature_mio };
        module::MIOS inference_outputs = { output_mio };
        auto inference_node = module::MNodeBuilderRegistry::getNode("SpeechCommands", configs, &inference_inputs,
            &inference_outputs, 2);
        if (!inference_node)
        {
            LOG(ERROR, "SpeechCommands node creation failed");
            return nullptr;
        }

        auto speech_commands_node = std::static_pointer_cast<algorithms::node::SpeechCommands>(inference_node);
        speech_commands_node->getModelOutputClasses();

        module::MIOS input_ios = { input_mio };
        auto input_node = module::MNodeBuilderRegistry::getNode("input", configs, &input_ios, &input_ios, 0);
        if (!input_node)
        {
            LOG(ERROR, "input node creation failed");
            return nullptr;
        }

        module::MIOS feature_inputs = { input_mio };
        module::MIOS feature_outputs = { feature_mio };
        auto feature_node = module::MNodeBuilderRegistry::getNode("SpeechCommandsPreprocess", configs, &feature_inputs,
            &feature_outputs, 1);
        if (!feature_node)
        {
            LOG(ERROR, "SpeechCommandsPreprocess node creation failed");
            return nullptr;
        }

        module::MIOS output_ios = { output_mio };
        auto output_node = module::MNodeBuilderRegistry::getNode("output", configs, &output_ios, &output_ios, 3);
        if (!output_node)
        {
            LOG(ERROR, "output node creation failed");
            return nullptr;
        }

        dag->addNode(input_node);
        dag->addNode(feature_node);
        dag->addNode(inference_node);
        dag->addNode(output_node);

        if (!dag->addEdge(input_node.get(), feature_node.get())
            || !dag->addEdge(feature_node.get(), inference_node.get())
            || !dag->addEdge(inference_node.get(), output_node.get()))
        {
            LOG(ERROR, "Failed to add edges to DAG");
            return nullptr;
        }

        return dag;
    }

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
