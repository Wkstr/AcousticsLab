#pragma once
#ifndef MODULE_NODE_HPP
#define MODULE_NODE_HPP

#include "module_io.hpp"

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <memory>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace module {

class MNode;

class MNodeBuilderRegistry final
{
public:
    using NodeBuilder = std::shared_ptr<MNode> (*)(int, MIOS &, MIOS &, const core::ConfigMap &);
    using NodeBuilderMap = std::unordered_map<std::string_view, NodeBuilder>;

    MNodeBuilderRegistry() = default;
    ~MNodeBuilderRegistry() = default;

    inline static std::shared_ptr<MNode> getNode(std::string_view name, int priority, MIOS &inputs, MIOS &outputs,
        const core::ConfigMap &configs) noexcept
    {
        auto it = _nodes.find(name);
        if (it != _nodes.end()) [[likely]]
        {
            return it->second(priority, inputs, outputs, configs);
        }
        return {};
    }

    static const NodeBuilderMap &getNodeBuilderMap() noexcept
    {
        return _nodes;
    }

    static core::Status registerNodeBuilder(std::string_view name, NodeBuilder builder, bool replace = false) noexcept;

private:
    static NodeBuilderMap _nodes;
};

class MNode
{
public:
    struct Lower final
    {
        inline bool operator()(const MNode *lhs, const MNode *rhs) const noexcept
        {
            const auto lhs_priority = lhs ? lhs->priority() : 0;
            const auto rhs_priority = rhs ? rhs->priority() : 0;
            return lhs_priority < rhs_priority;
        }
    };

    virtual ~MNode() = default;

    std::string_view name() const noexcept
    {
        return _name;
    }

    int priority() const noexcept
    {
        return _priority;
    }

    inline const MIOS &inputs() const noexcept
    {
        return _inputs;
    }

    inline const MIOS &outputs() const noexcept
    {
        return _outputs;
    }

    std::shared_ptr<MIO> input(size_t index) const noexcept
    {
        if (index < _inputs.size()) [[likely]]
        {
            return _inputs[index];
        }
        return {};
    }

    std::shared_ptr<MIO> input(std::string_view attribute) const noexcept
    {
        for (const auto &input: _inputs)
        {
            if (input->attribute == attribute)
            {
                return input;
            }
        }
        return {};
    }

    std::shared_ptr<MIO> output(size_t index) const noexcept
    {
        if (index < _outputs.size()) [[likely]]
        {
            return _outputs[index];
        }
        return {};
    }

    std::shared_ptr<MIO> output(std::string_view attribute) const noexcept
    {
        for (const auto &output: _outputs)
        {
            if (output->attribute == attribute)
            {
                return output;
            }
        }
        return {};
    }

    virtual core::Status config(const core::ConfigMap &configs) noexcept = 0;

    inline core::Status operator()() noexcept
    {
        return forward(_inputs, _outputs);
    }

protected:
    template<typename IS, typename OS>
    explicit MNode(std::string_view name, int priority, IS &&inputs, OS &&outputs) noexcept
        : _name(name), _priority(priority), _inputs(std::forward<IS>(inputs)), _outputs(std::forward<OS>(outputs))
    {
    }

    virtual inline core::Status forward(const MIOS &inputs, MIOS &outputs) noexcept = 0;

private:
    std::string_view _name;
    int _priority;
    MIOS _inputs;
    MIOS _outputs;
};

} // namespace module

namespace bridge {

extern void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__();
extern void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__();

} // namespace bridge

#endif
