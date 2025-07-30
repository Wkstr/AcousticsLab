#pragma once
#ifndef MODULE_IO_HPP
#define MODULE_IO_HPP

#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace module {

class MIO final
{
public:
    template<typename T,
        std::enable_if_t<std::is_same_v<std::remove_cvref_t<T>, std::shared_ptr<core::Tensor>>, bool> = true>
    constexpr explicit MIO(T &&tensor, std::string_view attribute = "") noexcept
        : _tensor(std::forward<T>(tensor)), _attribute(attribute)
    {
    }

    ~MIO() = default;

    constexpr inline core::Tensor *operator()() const noexcept
    {
        return _tensor.get();
    }

    constexpr inline std::string_view attribute() const noexcept
    {
        return _attribute;
    }

private:
    std::shared_ptr<core::Tensor> _tensor;
    const std::string_view _attribute;
};

using MIOS = std::vector<std::shared_ptr<MIO>>;

inline bool operator==(const MIOS &lhs, const MIOS &rhs) noexcept
{
    if (lhs.size() != rhs.size()) [[unlikely]]
    {
        return false;
    }
    return std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(),
        [](const std::shared_ptr<MIO> &a, const std::shared_ptr<MIO> &b) noexcept {
            return a && b && a->attribute() == b->attribute() && a->operator()() == b->operator()();
        });
}

} // namespace module

#endif
