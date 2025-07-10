#include "engine.hpp"

namespace hal {

core::Status EngineRegistry::registerEngine(Engine *engine) noexcept
{
    if (!engine)
    {
        return STATUS(EINVAL, "Engine cannot be null");
    }
    if (_engines.contains(engine->info().id))
    {
        return STATUS(EEXIST, "Engine with this ID already exists");
    }

    _engines[engine->info().id] = engine;

    return STATUS_OK();
}

EngineRegistry::EngineMap EngineRegistry::_engines = {};

} // namespace hal
