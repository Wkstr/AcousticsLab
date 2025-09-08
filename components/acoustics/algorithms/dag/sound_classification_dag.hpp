#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"

namespace algorithms { namespace dag {

    inline std::shared_ptr<module::MDAG> createSoundClassification(const core::ConfigMap &configs)
    {
        auto dag = std::make_shared<module::MDAG>("SoundClassification");

        auto inference_node = module::MNodeBuilderRegistry::getNode("SpeechCommands", configs, nullptr, nullptr, 2);
        if (!inference_node)
        {
            LOG(ERROR, "SpeechCommands node creation failed");
            return nullptr;
        }
        auto &inference_inputs = inference_node->inputs();
        if (inference_inputs.empty())
        {
            LOG(ERROR, "Inference node has no inputs after initialization");
            return nullptr;
        }
        auto feature_node
            = module::MNodeBuilderRegistry::getNode("SpeechCommandsPreprocess", configs, nullptr, &inference_inputs, 1);
        if (!feature_node)
        {
            LOG(ERROR, "SpeechCommandsPreprocess node creation failed");
            return nullptr;
        }
        auto &feature_inputs = feature_node->inputs();
        if (feature_inputs.empty())
        {
            LOG(ERROR, "Feature node has no inputs after initialization");
            return nullptr;
        }
        auto input_node = module::MNodeBuilderRegistry::getNode("input", configs, &feature_inputs, &feature_inputs, 0);
        if (!input_node)
        {
            LOG(ERROR, "input node creation failed");
            return nullptr;
        }
        auto &inference_outputs = inference_node->outputs();
        if (inference_outputs.empty())
        {
            LOG(ERROR, "Inference node has no outputs after initialization");
            return nullptr;
        }
        auto output_node
            = module::MNodeBuilderRegistry::getNode("output", configs, &inference_outputs, &inference_outputs, 3);
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

        LOG(INFO, "SoundClassification DAG created successfully");
        return dag;
    }

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
