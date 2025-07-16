#include "hal/engine.hpp"
#include "hal/engines/tflite_engine.hpp"
#include "hal/processor.hpp"
#include "hal/processors/feature_extractor.hpp"

namespace bridge {

void __REGISTER_PROCESSORS__()
{
    [[maybe_unused]] static hal::ProcessorFeatureExtractor processor_feature_extractor;
}

void __REGISTER_ENGINES__()
{
    [[maybe_unused]] static hal::EngineTFLite engine_tflite;
}

} // namespace bridge
