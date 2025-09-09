#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"

namespace algorithm { namespace dag {

    static inline std::shared_ptr<module::MDAG> createSoundClassification(const core::ConfigMap &configs)
    {
        auto inference_node = module::MNodeBuilderRegistry::getNode("SpeechCommands", configs, nullptr, nullptr);
        if (!inference_node)
        {
            return nullptr;
        }
        const auto &inference_inputs = inference_node->inputs();
        if (inference_inputs.empty())
        {
            return nullptr;
        }
        auto feature_node
            = module::MNodeBuilderRegistry::getNode("SpeechCommandsPreprocess", configs, nullptr, &inference_inputs);
        if (!feature_node)
        {
            return nullptr;
        }
        const auto &feature_inputs = feature_node->inputs();
        if (feature_inputs.empty())
        {
            return nullptr;
        }
        auto input_node = module::MNodeBuilderRegistry::getNode("input", configs, &feature_inputs, &feature_inputs);
        if (!input_node)
        {
            return nullptr;
        }
        const auto &inference_outputs = inference_node->outputs();
        if (inference_outputs.empty())
        {
            return nullptr;
        }
        auto output_node
            = module::MNodeBuilderRegistry::getNode("output", configs, &inference_outputs, &inference_outputs);
        if (!output_node)
        {
            return nullptr;
        }

        auto dag = std::make_shared<module::MDAG>("SoundClassification");
        if (!dag)
        {
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
            return nullptr;
        }

        LOG(DEBUG, "SoundClassification DAG created successfully");
        return dag;
    }

}} // namespace algorithm::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP