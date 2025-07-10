#pragma once
#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"

#include "hal/transport.hpp"

#include "context.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>
#include <vector>

namespace api {

class Executor;

class Task: public std::enable_shared_from_this<Task>
{
public:
    struct Lower final
    {
        inline bool operator()(const std::shared_ptr<Task> &lhs, const std::shared_ptr<Task> &rhs) const noexcept
        {
            const auto lhs_priority = lhs ? lhs->priority() : 0;
            const auto rhs_priority = rhs ? rhs->priority() : 0;
            return lhs_priority < rhs_priority;
        }
    };

    virtual ~Task() noexcept
    {
        LOG(DEBUG, "Task destroyed: ID=%zu, Priority=%zu", _id, _priority);
    }

    size_t id() const noexcept
    {
        return _id;
    }

    inline size_t priority() const noexcept
    {
        return _priority;
    }

    std::shared_ptr<Task> getptr() noexcept
    {
        return shared_from_this();
    }

    virtual core::Status operator()(Executor &executor) = 0;

protected:
    Task(Context &context, hal::Transport &transport, size_t id, size_t priority) noexcept
        : _context(context), _transport(transport), _id(id), _priority(priority)
    {
        LOG(DEBUG, "Task created: ID=%zu, Priority=%zu", _id, _priority);
    }

    Context &_context;
    hal::Transport &_transport;

private:
    const size_t _id;
    size_t _priority;
};

class Executor final
{
public:
    using TaskPtr = std::shared_ptr<Task>;
    using TaskQueue = std::priority_queue<TaskPtr, std::vector<TaskPtr>, Task::Lower>;
    using YieldCallback = void (*)();

    struct DelayedTask final
    {
        mutable TaskPtr task;
        std::chrono::steady_clock::time_point spawn_time;
        size_t delay_ms;

        inline bool isReady(std::chrono::steady_clock::time_point current_time) const noexcept
        {
            return std::chrono::duration_cast<std::chrono::milliseconds>(current_time - spawn_time).count() >= delay_ms;
        }
    };

    using DelayedTaskQueue = std::vector<DelayedTask>;

    Executor(size_t bound) noexcept;

    ~Executor();

    inline size_t id() const noexcept
    {
        return _id_counter.load();
    }

    inline core::Status submit(TaskPtr task, size_t delay_ms = 0) noexcept
    {
        if (!task)
        {
            return STATUS(ECANCELED, "Task is null");
        }

        if (task->id() != _id_counter.load()) [[unlikely]]
        {
            LOG(DEBUG, "Task cancelled or ID mismatch: Expected=%zu, Actual=%zu", _id_counter.load(), task->id());
            return STATUS(ECANCELED, "Task ID mismatch");
        }

        if (delay_ms > 0) [[unlikely]]
        {
            std::lock_guard<std::mutex> lock(_delayed_tasks_mutex);
            if (_delayed_tasks.size() >= _bound) [[unlikely]]
            {
                return STATUS(ENOSPC, "Executor delayed task queue is full");
            }
            _delayed_tasks.emplace_back(DelayedTask { task, std::chrono::steady_clock::now(), delay_ms });
        }
        else [[likely]]
        {
            std::lock_guard<std::mutex> lock(_mutex);
            if (_queue.size() >= _bound) [[unlikely]]
            {
                return STATUS(ENOSPC, "Executor queue is full");
            }
            _queue.push(task);
        }

        LOG(VERBOSE, "Task submitted: ID=%zu, Priority=%zu", task->id(), task->priority());

        return STATUS_OK();
    }

    inline core::Status execute(YieldCallback yield_callback = nullptr) noexcept
    {
        while (true)
        {
            TaskPtr task;

            {
                std::lock_guard<std::mutex> lock(_mutex);
                {
                    std::lock_guard<std::mutex> lock(_delayed_tasks_mutex);
                    std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
                    std::erase_if(_delayed_tasks, [&](const DelayedTask &dt) {
                        if (dt.isReady(current_time))
                        {
                            _queue.emplace(std::move(dt.task));
                            return true;
                        }
                        return false;
                    });
                }
                if (_queue.empty()) [[unlikely]]
                {
                    return STATUS_OK();
                }
                task = std::move(_queue.top());
                _queue.pop();
            }

            const size_t task_id = task->id();
            if (task_id != _id_counter.load()) [[unlikely]]
            {
                LOG(DEBUG, "Task cancelled or ID mismatch: Expected=%zu, Actual=%zu", _id_counter.load(), task_id);
                continue;
            }

            LOG(VERBOSE, "Processing task: ID=%zu, Priority=%zu", task_id, task->priority());

            const auto status = task->operator()(*this);
            if (!status) [[unlikely]]
            {
                LOG(ERROR, "Task execution failed: %d, %s", status.code(), status.message().c_str());
                return status;
            }
            if (yield_callback)
            {
                yield_callback();
            }
        }
    }

    inline size_t clear() noexcept
    {
        size_t cancelled_count = 0;
        _id_counter.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(_delayed_tasks_mutex);
            cancelled_count += _delayed_tasks.size();
            _delayed_tasks.clear();
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            cancelled_count += _queue.size();
            while (!_queue.empty())
            {
                _queue.pop();
            }
        }

        LOG(DEBUG, "Cancelled %zu tasks, current ID is now %zu", cancelled_count, _id_counter.load());

        return cancelled_count;
    }

private:
    const size_t _bound;
    std::atomic<size_t> _id_counter;
    TaskQueue _queue;
    std::mutex _mutex;
    DelayedTaskQueue _delayed_tasks;
    std::mutex _delayed_tasks_mutex;
};

} // namespace api

#endif // EXECUTOR_HPP
