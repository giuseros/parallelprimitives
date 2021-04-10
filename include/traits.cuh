#ifndef __TRAITS_CUH__
#define __TRAITS_CUH__

template <typename T, int vec_size>
struct vector_t;

template <>
struct vector_t<double, 2>
{
	using vec_type = double2;
};

template <>
struct vector_t<double, 4>
{
	using vec_type = double4;
};

template <>
struct vector_t<float, 2>
{
	using vec_type = float2;
};

template <>
struct vector_t<float, 4>
{
	using vec_type = float4;
};

template <>
struct vector_t<int, 2>
{
	using vec_type = int2;
};

template <>
struct vector_t<int, 4>
{
	using vec_type = int4;
};

template<typename T>
using vector_2 = typename vector_t<T, 2>::vec_type;

template<typename T>
using vector_4 = typename vector_t<T, 4>::vec_type;

#endif
