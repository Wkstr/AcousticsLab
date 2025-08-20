#pragma once
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "logger.hpp"
#include "types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace core {

class Tensor final
{
public:
    enum class Type : uint16_t {
        Unknown = 0,
        Int8 = (sizeof(int8_t) << 8) | 1,
        Int16 = (sizeof(int16_t) << 8) | 2,
        Int32 = (sizeof(int32_t) << 8) | 3,
        Int64 = (sizeof(int64_t) << 8) | 4,
        UInt8 = (sizeof(uint8_t) << 8) | 5,
        UInt16 = (sizeof(uint16_t) << 8) | 6,
        UInt32 = (sizeof(uint32_t) << 8) | 7,
        UInt64 = (sizeof(uint64_t) << 8) | 8,
        Float32 = (sizeof(float) << 8) | 9,
        Float64 = (sizeof(double) << 8) | 10,
        Class = (sizeof(class_t) << 8) | 11,
    };

    class Shape final
    {
    public:
        template<typename... Args,
            std::enable_if_t<(std::is_integral_v<std::remove_cvref_t<Args> > && ...), bool> = true>
        Shape(Args &&...dims) noexcept
            : _dims({ std::forward<Args>(dims)... }), _size(sizeof...(Args)), _dot((std::forward<Args>(dims) * ...))
        {
            _dims.shrink_to_fit();
        }

        explicit Shape(const std::vector<int> &dims) noexcept
            : _dims(dims), _size(dims.size()), _dot(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()))
        {
            _dims.shrink_to_fit();
        }

        explicit Shape(std::vector<int> &&dims) noexcept
            : _dims(std::move(dims)), _size(_dims.size()),
              _dot(std::accumulate(_dims.begin(), _dims.end(), 1, std::multiplies<>()))
        {
            _dims.shrink_to_fit();
        }

        ~Shape() = default;

        inline size_t size() const noexcept
        {
            return _size;
        }

        inline int operator[](size_t index) const noexcept
        {
            if (index >= _size) [[unlikely]]
            {
                return 0;
            }
            return _dims[index];
        }

        inline bool operator==(const Shape &other) const noexcept
        {
            return _dot == other._dot && std::equal(_dims.cbegin(), _dims.cend(), other._dims.cbegin());
        }

        inline int dot() const noexcept
        {
            return _dot;
        }

    private:
        std::vector<int> _dims;
        size_t _size;
        int _dot;
    };

    class QuantParams final
    {
    public:
        explicit constexpr QuantParams(float scale = 1.0f, int32_t zero_point = 0) noexcept
            : _scale(scale), _zero_point(zero_point)
        {
            if (std::isnan(_scale) || std::isinf(_scale) || _scale <= std::numeric_limits<float>::epsilon())
                [[unlikely]]
            {
                LOG(ERROR, "Invalid scale value: %f", static_cast<double>(_scale));
            }
        }

        ~QuantParams() = default;

        constexpr inline float scale() const noexcept
        {
            return _scale;
        }

        constexpr inline int32_t zeroPoint() const noexcept
        {
            return _zero_point;
        }

    private:
        float _scale;
        int32_t _zero_point;
    };

    template<typename T = std::unique_ptr<Tensor>, typename P>
    [[nodiscard]] static T create(Type dtype, P &&shape, std::shared_ptr<std::byte[]> data = nullptr,
        size_t size = 0) noexcept
    {
        if (dtype == Type::Unknown) [[unlikely]]
        {
            LOG(ERROR, "Tensor data type is unknown");
            return {};
        }
        const size_t dsize = static_cast<size_t>((static_cast<uint16_t>(dtype) >> 8));
        if (dsize == 0) [[unlikely]]
        {
            LOG(ERROR, "Data type size is zero for dtype: %d", static_cast<int>(dtype));
            return {};
        }

        if (shape.size() == 0) [[unlikely]]
        {
            LOG(ERROR, "Tensor shape cannot be empty");
            return {};
        }

        if (size && !data) [[unlikely]]
        {
            LOG(ERROR, "Data cannot be null if size is provided");
            return {};
        }

        const size_t bytes_required = dsize * shape.dot();
        if (data && size < bytes_required) [[unlikely]]
        {
            LOG(ERROR, "Data buffer size is too small: %zu bytes required, %zu bytes provided", bytes_required, size);
            return {};
        }
        if (!data)
        {
            data = std::shared_ptr<std::byte[]>(new (std::nothrow) std::byte[bytes_required]);
            if (!data) [[unlikely]]
            {
                LOG(ERROR, "Failed to allocate memory for tensor data, size: %zu bytes", bytes_required);
                return {};
            }
        }

        T ptr { new (std::nothrow) Tensor(dtype, dsize, std::forward<P>(shape), std::move(data), bytes_required) };
        if (!ptr) [[unlikely]]
        {
            LOG(ERROR, "Failed to create Tensor, size: %zu bytes", sizeof(Tensor));
            return {};
        }

        return ptr;
    }

    ~Tensor() = default;

    inline Type dtype() const noexcept
    {
        return _dtype;
    }

    inline size_t dsize() const noexcept
    {
        return _dsize;
    }

    inline size_t size() const noexcept
    {
        return _size;
    }

    inline const Shape &shape() const noexcept
    {
        return _shape;
    }

    inline std::shared_ptr<std::byte[]> data() noexcept
    {
        return _data;
    }

    template<typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
    inline T *data() noexcept
    {
        if (sizeof(T) != _dsize) [[unlikely]]
        {
            LOG(ERROR, "Data type mismatch: expected %zu bytes, got %zu bytes", sizeof(T),
                static_cast<size_t>((static_cast<uint16_t>(_dtype) >> 8)));
            return nullptr;
        }
        return reinterpret_cast<T *>(_data.get());
    }

    template<typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
    inline const T *data() const noexcept
    {
        return data<T>();
    }

    template<typename T, std::enable_if_t<std::is_trivially_copyable_v<T>, bool> = true>
    inline T operator[](size_t index) const noexcept
    {
        return reinterpret_cast<const T *>(_data.get())[index];
    }

    template<typename... Args>
    bool reshape(Args &&...dims) noexcept
    {
        return reshape(Shape(std::forward<Args>(dims)...));
    }

    bool reshape(const Shape &new_shape) noexcept
    {
        if (new_shape.size() == 0) [[unlikely]]
        {
            LOG(ERROR, "Invalid shape for reshaping, size: %zu", new_shape.size());
            return false;
        }
        if (_dsize * new_shape.dot() > _size) [[unlikely]]
        {
            LOG(ERROR, "New shape requires more data than available, required: %zu bytes, available: %zu bytes",
                _dsize * new_shape.dot(), _size);
            return false;
        }

        _shape = std::move(new_shape);

        return true;
    }

protected:
    template<typename T>
    explicit constexpr Tensor(Type dtype, size_t dsize, T &&shape, std::shared_ptr<std::byte[]> &&data,
        size_t size) noexcept
        : _dtype(dtype), _dsize(dsize), _shape(std::forward<T>(shape)), _data(std::move(data)), _size(size)
    {
    }

private:
    const Type _dtype;
    const size_t _dsize;
    Shape _shape;
    std::shared_ptr<std::byte[]> _data;
    const size_t _size;
};

} // namespace core

#endif // TENSOR_HPP
