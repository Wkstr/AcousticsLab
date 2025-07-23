#pragma once
#ifndef MODULE_IO_HPP
#define MODULE_IO_HPP

#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace module {

class MIO final
{
public:
    struct IO final
    {
        template<typename T,
            std::enable_if_t<std::is_same_v<std::remove_cvref_t<T>, std::shared_ptr<core::Tensor>>, bool> = true>
        explicit IO(T &&tensor, std::string_view attribute = "") noexcept
            : tensor(std::forward<T>(tensor)), attribute(attribute)
        {
        }

        ~IO() = default;

        std::shared_ptr<core::Tensor> tensor;
        const std::string_view attribute;
    };

    using IOS = std::vector<std::unique_ptr<IO>>;

    template<typename T = std::shared_ptr<MIO>, typename... Args>
    [[nodiscard]] static T create(Args &&...args) noexcept
    {
        return T { new MIO(IOS { std::make_unique<IO>(std::forward<Args>(args)...)... }) };
    }

    ~MIO() = default;

    const IOS &operator()() const noexcept
    {
        return _ios;
    }

private:
    explicit MIO(IOS &&ios) noexcept : _ios(std::move(ios))
    {
        if (_ios.empty()) [[unlikely]]
        {
            LOG(WARNING, "Module IO is empty");
        }
        else
        {
            _ios.shrink_to_fit();
        }
    }

    IOS _ios;
};

} // namespace module

#endif
