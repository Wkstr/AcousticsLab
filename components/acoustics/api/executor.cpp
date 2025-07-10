#include "executor.hpp"

namespace api {

Executor::Executor(size_t bound) noexcept : _bound(bound), _id_counter(0), _queue(), _mutex(), _delayed_tasks()
{
    if (bound == 0) [[unlikely]]
    {
        LOG(ERROR, "Executor bound cannot be zero");
    }
    LOG(INFO, "Executor initialized with bound: %zu", _bound);
}

Executor::~Executor()
{
    LOG(INFO, "Executor destroyed");
}

} // namespace api
