#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"
#include "module/node/node_input.hpp"
#include "module/node/node_output.hpp"

#include <esp_heap_caps.h>
#include <memory>

namespace algorithms { namespace dag {
    template<typename T>
    inline std::shared_ptr<core::Tensor> allocateAlignedTensor(const core::Tensor::Shape &shape,
        core::Tensor::Type type)
    {
        int total_elements = shape.dot();
        size_t total_bytes = static_cast<size_t>(total_elements) * sizeof(T);

        void *raw_data = heap_caps_aligned_alloc(16, total_bytes, MALLOC_CAP_SPIRAM);
        if (!raw_data)
        {
            LOG(ERROR, "Failed to allocate %zu bytes of aligned memory", total_bytes);
            return nullptr;
        }

        std::shared_ptr<std::byte[]> data(reinterpret_cast<std::byte *>(raw_data), [](std::byte *ptr) {
            if (ptr)
            {
                heap_caps_free(ptr);
            }
        });

        return core::Tensor::create<std::shared_ptr<core::Tensor>>(type, shape, data, total_bytes);
    }

    inline std::shared_ptr<module::MDAG> createSoundClassification(const core::ConfigMap &configs)
    {
        LOG(INFO, "Creating SoundClassification with dynamic tensor allocation");
        auto dag = std::make_shared<module::MDAG>("SoundClassification");

        constexpr int AUDIO_SAMPLES = 44032;
        constexpr int FEATURE_COUNT = 9976;
        constexpr int DEFAULT_OUTPUT_CLASSES = 20;

        auto input_tensor
            = allocateAlignedTensor<int16_t>(core::Tensor::Shape(AUDIO_SAMPLES), core::Tensor::Type::Int16);
        auto feature_tensor
            = allocateAlignedTensor<float>(core::Tensor::Shape(FEATURE_COUNT), core::Tensor::Type::Float32);
        auto output_tensor = core::Tensor::create<std::shared_ptr<core::Tensor>>(core::Tensor::Type::Class,
            core::Tensor::Shape(1, DEFAULT_OUTPUT_CLASSES));

        if (!input_tensor || !feature_tensor || !output_tensor)
        {
            LOG(ERROR, "Failed to create tensors");
            return nullptr;
        }

        auto input_mio = std::make_shared<module::MIO>(input_tensor, "audio_input");
        auto feature_mio = std::make_shared<module::MIO>(feature_tensor, "feature_data");
        auto output_mio = std::make_shared<module::MIO>(output_tensor, "classification_output");

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

        module::MIOS inference_inputs = { feature_mio };
        module::MIOS inference_outputs = { output_mio };
        auto inference_node = module::MNodeBuilderRegistry::getNode("SpeechCommands", configs, &inference_inputs,
            &inference_outputs, 2);
        if (!inference_node)
        {
            LOG(ERROR, "SpeechCommands node creation failed");
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

        LOG(INFO, "SoundClassification created successfully with default output shape (1, %d)", DEFAULT_OUTPUT_CLASSES);
        return dag;
    }

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
