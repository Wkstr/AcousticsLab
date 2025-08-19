#pragma once
#ifndef SOUND_CLASSIFICATION_DAG_HPP
#define SOUND_CLASSIFICATION_DAG_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "module/module_dag.hpp"

#include <memory>
#include <string>

namespace algorithms { namespace dag {

    class SoundClassificationDAG
    {
    public:
        explicit SoundClassificationDAG(std::string_view name) noexcept : _dag(std::make_shared<module::MDAG>(name)) { }

        module::MNode *addNode(std::shared_ptr<module::MNode> node) noexcept
        {
            return _dag->addNode(node);
        }

        bool addEdge(module::MNode *from, module::MNode *to) noexcept
        {
            return _dag->addEdge(from, to);
        }

        module::MNode *node(std::string_view name) const noexcept
        {
            return _dag->node(name);
        }

        const std::forward_list<std::shared_ptr<module::MNode>> &nodes() const noexcept
        {
            return _dag->nodes();
        }

        core::Status operator()() noexcept
        {
            return _dag->operator()();
        }

        core::Status operator()(core::Reporter &reporter) noexcept
        {
            return _dag->operator()(reporter);
        }

        std::shared_ptr<module::MDAG> getDAG() const noexcept
        {
            return _dag;
        }

    private:
        std::shared_ptr<module::MDAG> _dag;
    };

    class SoundClassificationDAGBuilder
    {
    public:
        static std::shared_ptr<module::MDAG> create(const core::ConfigMap &configs);
        static constexpr const char *FEATURE_EXTRACTOR_NODE_NAME = "ESPFeatureExtractorNode";
        static constexpr const char *INFERENCE_NODE_NAME = "SpeechCommandsNode";

    private:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FEATURE_COUNT = 9976;

        static std::shared_ptr<core::Tensor> createInputTensor();

        static std::shared_ptr<core::Tensor> createFeatureTensor();

        static std::shared_ptr<module::MNode> createFeatureExtractorNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        static std::shared_ptr<module::MNode> createInferenceNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        template<typename T>
        static std::shared_ptr<core::Tensor> allocateAlignedTensor(const core::Tensor::Shape &shape,
            core::Tensor::Type type);

        static core::Status validateConfig(const core::ConfigMap &configs);
    };

    std::shared_ptr<module::MDAG> createSoundClassificationDAG(const core::ConfigMap &configs);

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
