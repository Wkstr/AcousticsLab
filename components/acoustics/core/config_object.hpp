#pragma once
#ifndef CONFIG_OBJECT_HPP
#define CONFIG_OBJECT_HPP

#include "logger.hpp"
#include "status.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace core {

class ConfigObject final
{
protected:
    struct Integer final
    {
        int value;
        int default_value;
        int min_value;
        int max_value;

        Integer(int default_value, int min_value, int max_value)
            : value(default_value), default_value(default_value), min_value(min_value), max_value(max_value)
        {
            if (min_value > max_value)
            {
                LOG(ERROR, "Invalid range for Integer: min_value (%d) > max_value (%d)", min_value, max_value);
            }
            if (default_value < min_value || default_value > max_value)
            {
                LOG(ERROR, "Default value (%d) out of range [%d, %d]", default_value, min_value, max_value);
                value = default_value = min_value;
            }
        }

        ~Integer() = default;

        bool validAndSet(const int &new_value)
        {
            if (new_value < min_value || new_value > max_value)
            {
                return false;
            }
            value = new_value;
            return true;
        }

        template<typename T>
        bool convertAndSet(const T &new_value)
        {
            if constexpr (std::is_integral_v<T>)
            {
                return validAndSet(static_cast<int>(new_value));
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                return validAndSet(static_cast<int>(std::round(new_value)));
            }
            else if constexpr (std::is_same_v<T, std::string>)
            {
                LOG(DEBUG, "Converting string to integer: '%s'", new_value.c_str());

                if (new_value.empty()) [[unlikely]]
                {
                    LOG(DEBUG, "String is empty, cannot convert to integer");
                    return false;
                }

                size_t dot_count = 0;
                const size_t size = new_value.size();
                for (size_t i = new_value[0] == '-' ? 1 : 0; i < size; ++i)
                {
                    const char c = new_value[i];
                    if (c == '.') [[unlikely]]
                    {
                        if (++dot_count > 1) [[unlikely]]
                        {
                            LOG(DEBUG, "More than one dot in string, cannot convert to integer");
                            return false;
                        }
                        continue;
                    }
                    else if (!std::isdigit(c))
                    {
                        return false;
                    }
                }

                if (dot_count)
                {
                    return validAndSet(static_cast<int>(std::round(std::atof(new_value.c_str()))));
                }
                else
                {
                    return validAndSet(std::atoi(new_value.c_str()));
                }
            }

            LOG(WARNING, "Failed to convert value to Integer");

            return false;
        }
    };

    struct Float final
    {
        float value;
        float default_value;
        float min_value;
        float max_value;

        Float(float default_value, float min_value, float max_value)
            : value(default_value), default_value(default_value), min_value(min_value), max_value(max_value)
        {
            if (min_value > max_value)
            {
                LOG(ERROR, "Invalid range for Float: min_value (%f) > max_value (%f)", static_cast<double>(min_value),
                    static_cast<double>(max_value));
            }
            if (default_value < min_value || default_value > max_value)
            {
                LOG(ERROR, "Default value (%f) out of range [%f, %f]", static_cast<double>(default_value),
                    static_cast<double>(min_value), static_cast<double>(max_value));
                value = default_value = min_value;
            }
        }

        ~Float() = default;

        bool validAndSet(float new_value)
        {
            if (new_value < min_value || new_value > max_value)
            {
                return false;
            }
            value = new_value;
            return true;
        }

        template<typename T>
        bool convertAndSet(const T &new_value)
        {
            if constexpr (std::is_integral_v<T>)
            {
                return validAndSet(static_cast<float>(new_value));
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                return validAndSet(static_cast<float>(new_value));
            }
            else if constexpr (std::is_same_v<T, std::string>)
            {
                LOG(DEBUG, "Converting string to float: '%s'", new_value.c_str());

                if (new_value.empty()) [[unlikely]]
                {
                    LOG(DEBUG, "String is empty, cannot convert to float");
                    return false;
                }

                const size_t size = new_value.size();
                for (size_t i = new_value[0] == '-' ? 1 : 0, dot_count = 0; i < size; ++i)
                {
                    const char c = new_value[i];
                    if (c == '.') [[unlikely]]
                    {
                        if (++dot_count > 1) [[unlikely]]
                        {
                            LOG(DEBUG, "More than one dot in string, cannot convert to float");
                            return false;
                        }
                        continue;
                    }
                    else if (!std::isdigit(c))
                    {
                        return false;
                    }
                }

                return validAndSet(std::atof(new_value.c_str()));
            }

            LOG(WARNING, "Failed to convert value to Float");

            return false;
        }
    };

    struct Boolean final
    {
        bool value;
        bool default_value;

        Boolean(bool default_value) : value(default_value), default_value(default_value) { }

        ~Boolean() = default;

        bool validAndSet(bool new_value)
        {
            value = new_value;
            return true;
        }

        template<typename T>
        bool convertAndSet(const T &new_value)
        {
            if constexpr (std::is_same_v<T, bool>)
            {
                return validAndSet(new_value);
            }
            else if constexpr (std::is_integral_v<T>)
            {
                return validAndSet(static_cast<bool>(new_value));
            }
            else if constexpr (std::is_floating_point_v<T>)
            {
                return validAndSet(static_cast<bool>(new_value));
            }
            else if constexpr (std::is_same_v<T, std::string>)
            {
                if (new_value == "true" || new_value == "1")
                {
                    return validAndSet(true);
                }
                else if (new_value == "false" || new_value == "0")
                {
                    return validAndSet(false);
                }
            }

            LOG(WARNING, "Failed to convert value to Boolean");

            return false;
        }
    };

    struct String final
    {
        std::string value;
        std::string default_value;

        String(const std::string &default_value) : value(default_value), default_value(default_value) { }

        ~String() = default;

        bool validAndSet(const std::string &new_value)
        {
            value = new_value;
            return true;
        }

        template<typename T>
        bool convertAndSet(const T &new_value)
        {
            if constexpr (std::is_same_v<T, std::string>)
            {
                return validAndSet(new_value);
            }

            LOG(WARNING, "Failed to convert value to String");

            return false;
        }
    };

public:
    enum class Type { Unknown, Integer, Float, Boolean, String };

    ConfigObject() : _name(""), _description(""), _type(Type::Unknown), _value(Integer(0, INT32_MIN, INT32_MAX))
    {
        LOG(ERROR, "Default constructor called for ConfigObject, this should not happen");
    }

    ~ConfigObject() = default;

    static ConfigObject createInteger(const std::string &name, const std::string &description, int default_value = 0,
        int min_value = INT32_MIN, int max_value = INT32_MAX)
    {
        return ConfigObject { name, description, Integer(default_value, min_value, max_value) };
    }

    static ConfigObject createFloat(const std::string &name, const std::string &description, float default_value = 0.0f,
        float min_value = -std::numeric_limits<float>::infinity(),
        float max_value = std::numeric_limits<float>::infinity())
    {
        return ConfigObject { name, description, Float(default_value, min_value, max_value) };
    }

    static ConfigObject createBoolean(const std::string &name, const std::string &description,
        bool default_value = false)
    {
        return ConfigObject { name, description, Boolean(default_value) };
    }

    static ConfigObject createString(const std::string &name, const std::string &description,
        const std::string &default_value = "")
    {
        return ConfigObject { name, description, String(default_value) };
    }

    template<typename T>
    Status setValue(const T &new_value)
    {
        using namespace std::string_literals;

        return std::visit(
            [&new_value, this]<typename P>(P &v) -> Status {
                using ValueType = P;
                using NewValueType = std::decay_t<std::remove_all_extents_t<T>>;
                if constexpr (std::is_same_v<ValueType, NewValueType>)
                {
                    if (!v.validAndSet(new_value))
                    {
                        return Status(EINVAL, "Value out of range for "s + this->_name);
                    }
                }
                else
                {
                    if (!v.convertAndSet(new_value))
                    {
                        return Status(EINVAL, "Type mismatch or value out of range for "s + this->_name);
                    }
                }
                return Status::OK();
            },
            _value);
    }

    template<typename T>
    inline T getValue(T default_value = T {}) const
    {
        using namespace std::string_literals;

        if (_type == Type::Unknown)
        {
            LOG(ERROR, "ConfigObject '%s' has unknown type", _name.c_str());
            return default_value;
        }

        return std::visit(
            [&](const auto &v) -> T {
                using ValueType = std::decay_t<decltype(v.value)>;
                if constexpr (std::is_same_v<ValueType, T>)
                {
                    return v.value;
                }
                else
                {
                    return default_value;
                }
            },
            _value);
    }

    const std::string &name() const
    {
        return _name;
    }

    const std::string &description() const
    {
        return _description;
    }

    Type type() const
    {
        return _type;
    }

    void resetToDefault()
    {
        std::visit([](auto &v) { v.value = v.default_value; }, _value);
    }

protected:
    ConfigObject(const std::string &name, const std::string &description,
        std::variant<Integer, Float, Boolean, String> &&value)
        : _name(name), _description(description), _type(Type::Unknown), _value(std::move(value))
    {
        if (std::holds_alternative<Integer>(_value))
        {
            _type = Type::Integer;
        }
        else if (std::holds_alternative<Float>(_value))
        {
            _type = Type::Float;
        }
        else if (std::holds_alternative<Boolean>(_value))
        {
            _type = Type::Boolean;
        }
        else if (std::holds_alternative<String>(_value))
        {
            _type = Type::String;
        }
    }

private:
    std::string _name;
    std::string _description;
    Type _type;

    std::variant<Integer, Float, Boolean, String> _value;
};

using ConfigObjectMap = std::unordered_map<std::string_view, ConfigObject>;

using ConfigMap = std::unordered_map<std::string_view, std::variant<int, float, bool, std::string>>;

ConfigMap operator&(const ConfigMap &lhs, const ConfigMap &rhs);

ConfigMap operator^(const ConfigMap &lhs, const ConfigMap &rhs);

ConfigMap fromConfigObjectMap(const ConfigObjectMap &config_objects);

ConfigMap fromConfigMap(const ConfigMap &config_map, const ConfigObjectMap &config_objects);

Status updateConfigObjectMap(ConfigObjectMap &config_objects, const ConfigMap &config_map);

#if defined(CONFIG_OBJECT_DECL_INTEGER)
#warning "Removing existing CONFIG_OBJECT_DECL_INTEGER"
#undef CONFIG_OBJECT_DECL_INTEGER
#endif
#define CONFIG_OBJECT_DECL_INTEGER(name, description, default_value, min, max)                                         \
    { name, core::ConfigObject::createInteger(name, description, default_value, min, max) }

#if defined(CONFIG_OBJECT_DECL_FLOAT)
#warning "Removing existing CONFIG_OBJECT_DECL_FLOAT"
#undef CONFIG_OBJECT_DECL_FLOAT
#endif
#define CONFIG_OBJECT_DECL_FLOAT(name, description, default_value, min, max)                                           \
    { name, core::ConfigObject::createFloat(name, description, default_value, min, max) }

#if defined(CONFIG_OBJECT_DECL_BOOLEAN)
#warning "Removing existing CONFIG_OBJECT_DECL_BOOLEAN"
#undef CONFIG_OBJECT_DECL_BOOLEAN
#endif
#define CONFIG_OBJECT_DECL_BOOLEAN(name, description, default_value)                                                   \
    { name, core::ConfigObject::createBoolean(name, description, default_value) }

#if defined(CONFIG_OBJECT_DECL_STRING)
#warning "Removing existing CONFIG_OBJECT_DECL_STRING"
#undef CONFIG_OBJECT_DECL_STRING
#endif
#define CONFIG_OBJECT_DECL_STRING(name, description, default_value)                                                    \
    { name, core::ConfigObject::createString(name, description, default_value) }

} // namespace core

#endif
