#include "config_object.hpp"

namespace core {

ConfigMap operator&(const ConfigMap &lhs, const ConfigMap &rhs)
{
    ConfigMap result = lhs;

    std::erase_if(result, [&](const auto &pair) {
        const auto &key = pair.first;
        return !rhs.contains(key);
    });

    return result;
}

ConfigMap operator^(const ConfigMap &lhs, const ConfigMap &rhs)
{
    ConfigMap result = lhs;

    std::erase_if(result, [&](const auto &pair) {
        const auto &key = pair.first;
        const auto &rhs_value = rhs.find(key);
        if (rhs_value == rhs.end())
        {
            return false;
        }
        const auto &lhs_value = pair.second;
        const auto &rhs_value_variant = rhs_value->second;
        return lhs_value == rhs_value_variant;
    });
    for (const auto &pair: rhs)
    {
        const auto &key = pair.first;
        if (!result.contains(key))
        {
            result.emplace(key, pair.second);
        }
    }

    return result;
}

ConfigMap fromConfigObjectMap(const ConfigObjectMap &config_objects)
{
    ConfigMap config_map;
    for (const auto &[key, config]: config_objects)
    {
        switch (config.type())
        {
            case ConfigObject::Type::Integer:
                config_map[key] = config.getValue<int>();
                break;
            case ConfigObject::Type::Float:
                config_map[key] = config.getValue<float>();
                break;
            case ConfigObject::Type::Boolean:
                config_map[key] = config.getValue<bool>();
                break;
            case ConfigObject::Type::String:
                config_map[key] = config.getValue<std::string>();
                break;
            default:
                LOG(ERROR, "Unknown ConfigObject type for key: %s", key.data());
                break;
        }
    }
    return config_map;
}

ConfigMap fromConfigMap(const ConfigMap &config_map, const ConfigObjectMap &config_objects)
{
    ConfigMap result = config_map;

    std::erase_if(result, [&](const auto &pair) {
        const auto &key = pair.first;
        return !config_objects.contains(key);
    });

    return result;
}

Status updateConfigObjectMap(ConfigObjectMap &config_objects, const ConfigMap &config_map)
{
    auto target = fromConfigObjectMap(config_objects);
    target = target ^ config_map;
    for (const auto &[key, value]: target)
    {
        auto it = config_objects.find(key);
        if (it != config_objects.end())
        {
            auto status = it->second.setValue(value);
            if (!status)
            {
                LOG(ERROR, "Failed to set value for config '%s': %s", key.data(), status.message().c_str());
                return status;
            }
            continue;
        }
        LOG(WARNING, "Config '%s' not found in ConfigObjectMap", key.data());
    }
    return Status::OK();
}

} // namespace core
