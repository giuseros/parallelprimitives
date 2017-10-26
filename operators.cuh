#ifndef OPERATORS_CUH__
#define OPERATORS_CUH__

#include "constants.cuh"

struct Plus{
	inline static __host__ __device__
	double apply(double a, double b)
	{
		return a+b;
	}

	inline static __device__
	double identity(){ return 0; }
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
		return -CUDART_INF;
	}

};

#endif
