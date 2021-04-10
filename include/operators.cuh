#ifndef OPERATORS_CUH__
#define OPERATORS_CUH__

#include "traits.cuh"

#define CUDART_INF_F __int_as_float(0x7f800000)
#define CUDART_INF_D  __longlong_as_double(0x7ff0000000000000ULL)

struct Plus{
	template<typename T>
	inline static __host__ __device__
	T apply(T a, T b)
	{
		return T(a+b);
	}

	template<typename T>
	inline static __host__ __device__
	T identity(T){ return T(0); }

	template <typename T>
	inline static __device__
	T apply_atomic(T *a_ptr, T b)
	{
		return atomicAdd(a_ptr, b);
	}
};

struct Max{
	inline static __host__ __device__
	double apply(double a, double b){
		if (a>b){
			return a;
		}else {
			return b;
		}
	}

	inline static __device__
	double identity(){
		return -CUDART_INF_D;
	}
};

template<int K, typename T>
struct Add{
	inline static __host__ __device__
	double apply(T a){
		return a+T(K);
	}
};

template <typename T>
struct LT{
	inline static __device__
	bool compare(T a, T b)
	{
		return a < b;
	}

	inline static __device__
	T inf()
	{
		return CUDART_INF_F;
	}
};

#endif
