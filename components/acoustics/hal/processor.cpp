#include "processor.hpp"

namespace hal {

core::Status ProcessorRegistry::registerProcessor(Processor *processor) noexcept
{
    if (!processor)
    {
        return STATUS(EINVAL, "Processor cannot be null");
    }
    if (_processors.contains(processor->info().id))
    {
        return STATUS(EEXIST, "Processor with this ID already exists");
    }

    _processors[processor->info().id] = processor;

    return STATUS_OK();
}

ProcessorRegistry::ProcessorMap ProcessorRegistry::_processors = {};

} // namespace hal
