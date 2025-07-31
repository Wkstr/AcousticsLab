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

    class SoundClassificationDAGBuilder
    {
    public:
        static std::shared_ptr<module::MDAG> create(const core::ConfigMap &configs);

    private:
        static constexpr size_t AUDIO_SAMPLES = 44032;
        static constexpr size_t FEATURE_COUNT = 9976;
        static constexpr size_t OUTPUT_CLASSES = 3;

        static std::shared_ptr<core::Tensor> createInputTensor();

        static std::shared_ptr<core::Tensor> createFeatureTensor();

        static std::shared_ptr<core::Tensor> createOutputTensor(size_t output_size = OUTPUT_CLASSES);

        static std::shared_ptr<module::MNode> createFeatureExtractorNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        static std::shared_ptr<module::MNode> createInferenceNode(const core::ConfigMap &configs,
            std::shared_ptr<module::MIO> input_mio, std::shared_ptr<module::MIO> output_mio);

        template<typename T>
        static std::shared_ptr<core::Tensor> allocateAlignedTensor(const core::Tensor::Shape &shape,
            core::Tensor::Type type);

        static size_t getOutputSize(const core::ConfigMap &configs);

        static core::Status validateConfig(const core::ConfigMap &configs);
    };

    std::shared_ptr<module::MDAG> createSoundClassificationDAG(const core::ConfigMap &configs);

}} // namespace algorithms::dag

#endif // SOUND_CLASSIFICATION_DAG_HPP
