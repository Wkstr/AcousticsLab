#include "hal/engine.hpp"

#include "engine/tflm.hpp"

namespace bridge {

void __REGISTER_ENGINES__()
{
#if defined(PORTING_LIB_TFLM_ENABLE) && PORTING_LIB_TFLM_ENABLE
    [[maybe_unused]] static porting::EngineTFLM engine_tflm;
#endif
}

} // namespace bridge
