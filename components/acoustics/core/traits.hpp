#pragma once
#ifndef TRAITS_HPP
#define TRAITS_HPP

#include <type_traits>
#include <utility>

namespace traits {

template<typename, typename = std::void_t<>>
struct is_bounded_char_array: std::false_type
{
};

template<typename T>
struct is_bounded_char_array<T, std::void_t<decltype(std::declval<T>()[0])>>
    : std::integral_constant<bool, std::is_same_v<std::remove_extent_t<T>, char>>
{
};

template<typename T>
inline constexpr bool is_bounded_char_array_v = is_bounded_char_array<T>::value;

template<typename, typename = std::void_t<>>
struct is_stl_container: std::false_type
{
};

template<typename T>
struct is_stl_container<T, std::void_t<typename T::iterator>>: std::true_type
{
};

template<typename T>
inline constexpr bool is_stl_container_v = is_stl_container<T>::value;

template<typename, typename = std::void_t<>>
struct is_stl_map_container: std::false_type
{
};

template<typename T>
struct is_stl_map_container<T, std::void_t<typename T::value_type, typename T::key_type, typename T::mapped_type>>
    : std::integral_constant<bool,
          std::is_same_v<typename T::value_type, std::pair<const typename T::key_type, typename T::mapped_type>>>
{
};

template<typename T>
inline constexpr bool is_stl_map_container_v = is_stl_map_container<T>::value;

template<typename>
struct is_variant: std::false_type
{
};

template<typename... Args>
struct is_variant<std::variant<Args...>>: std::true_type
{
};

template<typename T>
inline constexpr bool is_variant_v = is_variant<T>::value;

template<typename T, typename = std::void_t<>>
struct stl_container_value
{
    using type = void;
};

template<typename T>
struct stl_container_value<T, std::void_t<typename T::value_type>>
{
    using type = typename T::value_type;
};

template<typename T>
using stl_container_value_t = typename stl_container_value<T>::type;

} // namespace traits

#endif // TRAITS_HPP
