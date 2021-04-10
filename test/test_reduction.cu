#include "operators.cuh"
#include "gpuvector.hpp"
#include "reduce.cuh"

#include <vector>
#include <iostream>
#include <numeric>
using namespace std;
using namespace pp;

template <typename T>
void simple_test_reduction()
{
	int N = 10000;

	auto hX = std::vector<T>(N);

	for (int i = 0; i< N; i++){
		hX[i] = T(1);
	}
	auto dX = gpuVector<T>(hX);

	auto rgpu1 = reduce<Plus,float>(dX, ReductionMethod::warp_reduction);
	// std::cout<<rgpu1<<std::endl;
	auto dX1 = gpuVector<T>(hX);
	auto rgpu2 = reduce<Plus,float>(dX1, ReductionMethod::shared_mem_reduction);
	// std::cout<<rgpu2<<std::endl;
	auto rcpu = std::accumulate(hX.begin(), hX.end(), T(0));

	if (rgpu1 != rcpu)
	{
		std::cerr<<"Warp reduction failed with N "<<N<<": "<<rcpu<<"!="<<rgpu1<<std::endl;
		return;
	}
	if (rgpu2 != rcpu)
	{
		std::cerr<<"Shared mem reduction failed with N "<<N<<": "<<rcpu<<"!="<<rgpu2<<std::endl;
		return;
	}

	std::cout<<"Tests passed"<<std::endl;
}

int main()
{
	cudaDeviceProp prop;

	cudaSetDevice(0);

	cudaGetDeviceProperties(&prop, 0);
	std::cout<<prop.name<<std::endl;

	std::cout<<"Using device "<<0<<":"<<prop.name<<std::endl;

	simple_test_reduction<float>();


    return 0;
}
