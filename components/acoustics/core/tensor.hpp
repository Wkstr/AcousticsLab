#pragma once
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "logger.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory_resource>
#include <numeric>
#include <type_traits>
#include <utility>

namespace core {

static constexpr size_t TENSOR_SHAPE_MAX_DIMENSIONS = 4;

class Tensor final
{
public:
    enum class Type : uint16_t {
        Unknown = 0,
        Int8 = (1 << 8) | 1,
        Int16 = (2 << 8) | 2,
        Int32 = (4 << 8) | 3,
        Int64 = (8 << 8) | 4,
        UInt8 = (1 << 8) | 5,
        UInt16 = (2 << 8) | 6,
        UInt32 = (4 << 8) | 7,
        UInt64 = (8 << 8) | 8,
        Float32 = (4 << 8) | 9,
        Float64 = (8 << 8) | 10,
    };

    class Shape final
    {
    public:
        Shape(std::initializer_list<size_t> dimensions) : _dimensions { 0 }, _size(0)
        {
            if (dimensions.size() > TENSOR_SHAPE_MAX_DIMENSIONS)
            {
                LOG(ERROR, "Too many dimensions: %zu > %zu", dimensions.size(), TENSOR_SHAPE_MAX_DIMENSIONS);
                return;
            }

            _size = dimensions.size();
            for (size_t i = 0; i < _size; ++i)
            {
                const size_t dim = *std::next(dimensions.begin(), i);
                if (dim == 0)
                {
                    LOG(ERROR, "Dimension cannot be zero: %zu", dim);
                    return;
                }
                _dimensions[i] = dim;
            }
        }

        inline size_t size() const
        {
            return _size;
        }

        inline size_t operator[](size_t index) const
        {
            if (index >= _size)
            {
                LOG(ERROR, "Index out of bounds: %zu >= %zu", index, _size);
                return 0;
            }
            return _dimensions[index];
        }

        inline size_t dot() const
        {
            return std::accumulate(_dimensions.begin(), _dimensions.begin() + _size, 1, std::multiplies<size_t>());
        }

    private:
        std::array<size_t, TENSOR_SHAPE_MAX_DIMENSIONS> _dimensions;
        size_t _size;
    };

    struct QuantParams final
    {
        QuantParams(float scale = 1.0f, int32_t zero_point = 0) : scale(scale), zero_point(zero_point)
        {
            if (std::isnan(scale) || std::isinf(scale) || scale <= std::numeric_limits<float>::epsilon()) [[unlikely]]
            {
                LOG(ERROR, "Invalid scale value: %f", static_cast<double>(scale));
            }
        }

        ~QuantParams() = default;

        const float scale;
        const int32_t zero_point;
    };

    Tensor(Type dtype, std::initializer_list<size_t> shape, std::shared_ptr<std::byte[]> data = nullptr,
        size_t data_size = 0) noexcept
        : _dtype(dtype), _elem_bytes(static_cast<size_t>((static_cast<uint16_t>(_dtype) >> 8))), _shape(shape),
          _data(std::move(data)), _data_size(data_size)
    {
        if (_elem_bytes == 0)
        {
            LOG(ERROR, "Unknown data type: %d", static_cast<int>(_dtype));
            return;
        }

        if (_shape.size() == 0)
        {
            LOG(ERROR, "Shape cannot be empty");
            return;
        }

        _data_size = _elem_bytes * _shape.dot();

        if (_data == nullptr)
        {
            _data = std::make_unique<std::byte[]>(_data_size);
            if (_data == nullptr)
            {
                LOG(ERROR, "Failed to allocate memory for tensor data");
                return;
            }
            LOG(INFO, "Allocated %zu bytes for tensor data at %p", _data_size, _data.get());
            std::fill_n(_data.get(), _data_size, std::byte(0));
        }

        if (_data_size > data_size)
        {
            LOG(ERROR, "Data buffer size is too small: %zu bytes required, %zu bytes provided", _data_size, data_size);
            return;
        }
    }

    Tensor(const Tensor &other) = delete;

    Tensor(Tensor &&other) noexcept
        : _dtype(other._dtype), _elem_bytes(other._elem_bytes), _shape(std::move(other._shape)),
          _data(std::move(other._data)), _data_size(other._data_size)
    {
        other._dtype = Type::Unknown;
        other._elem_bytes = 0;
        other._shape = Shape({});
        other._data.reset();
        other._data_size = 0;
    }

    Tensor() noexcept : _dtype(Type::Unknown), _elem_bytes(0), _shape({}), _data(nullptr), _data_size(0) { }

    ~Tensor() = default;

    Tensor &operator=(const Tensor &other) = delete;

    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            _dtype = other._dtype;
            _elem_bytes = other._elem_bytes;
            _shape = std::move(other._shape);
            _data = std::move(other._data);
            _data_size = other._data_size;
        }
        return *this;
    }

    inline Type dtype() const
    {
        return _dtype;
    }

    inline const Shape &shape() const
    {
        return _shape;
    }

    inline size_t dataSize() const
    {
        return _data_size;
    }

    inline const std::byte *data() const
    {
        return _data.get();
    }

    template<typename T>
    inline T *dataAs()
    {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        if (sizeof(T) != _elem_bytes) [[unlikely]]
        {
            LOG(ERROR, "Data type mismatch: expected %zu bytes, got %zu bytes", sizeof(T),
                static_cast<size_t>((static_cast<uint16_t>(_dtype) >> 8)));
            return nullptr;
        }
        return reinterpret_cast<T *>(_data.get());
    }

    template<typename T>
    inline const T *dataAs() const
    {
        return reinterpret_cast<const T *>(_data.get());
    }

    template<typename T>
    inline T operator[](size_t index) const
    {
        return reinterpret_cast<const T *>(_data.get())[index];
    }

    bool reshape(std::initializer_list<size_t> new_shape)
    {
        Shape new_shape_obj(new_shape);
        if (new_shape_obj.size() == 0 || new_shape_obj.dot() == 0)
        {
            LOG(ERROR, "Invalid shape for reshaping");
            return false;
        }

        const size_t new_size = new_shape_obj.dot() * _elem_bytes;
        if (new_size != _data_size)
        {
            LOG(ERROR, "Reshape size mismatch: %zu bytes expected, %zu bytes available", new_size, _data_size);
            return false;
        }

        _shape = std::move(new_shape_obj);

        return true;
    }

private:
    Type _dtype;
    size_t _elem_bytes;
    Shape _shape;
    std::shared_ptr<std::byte[]> _data;
    size_t _data_size;
};

} // namespace core

#endif // TENSOR_HPP
