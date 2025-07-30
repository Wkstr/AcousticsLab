#pragma once
#ifndef MODULE_IO_HPP
#define MODULE_IO_HPP

#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace module {

struct MIO final
{
    template<typename T,
        std::enable_if_t<std::is_same_v<std::remove_cvref_t<T>, std::shared_ptr<core::Tensor>>, bool> = true>
    explicit MIO(T &&tensor, std::string_view attribute = "") noexcept
        : tensor(std::forward<T>(tensor)), attribute(attribute)
    {
    }

    ~MIO() = default;

    inline core::Tensor *operator()() const noexcept
    {
        return tensor.get();
    }

    std::shared_ptr<core::Tensor> tensor;
    const std::string_view attribute;
};

using MIOS = std::vector<std::shared_ptr<MIO>>;

bool operator==(const MIOS &lhs, const MIOS &rhs) noexcept
{
    if (lhs.size() != rhs.size()) [[unlikely]]
    {
        return false;
    }
    return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(),
        [](const std::shared_ptr<MIO> &a, const std::shared_ptr<MIO> &b) noexcept {
            return a && b && a->attribute == b->attribute && a->tensor == b->tensor;
        });
}

} // namespace module

#endif
