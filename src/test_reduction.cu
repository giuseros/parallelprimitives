#include "gpuvector.cuh"
#include "utils.cuh"
#include "perf.h"
#include "scan.cuh"
#include "operators.cuh"
#include "bulk.cuh"
#include "reduce.cuh"

#include <thrust/reduce.h>

#include <vector>
#include <iostream>
#include <numeric>
using namespace std;

template <typename T>
void simple_test_reduction()
{
	int N = 1e5;

	auto hX = std::vector<T>(N);

	for (int i = 0; i< N; i++){
		hX[i] = T(1);
	}
	auto dX = gpuVector<T>(hX);

	auto rgpu1 = reduce<Plus>(dX);
	auto rgpu2 = reduce<Plus>(dX, ReductionMethod::shared_mem_reduction);
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

template<typename T>
void run_reduction_matlab()
{
	std::vector<float> gpu_algo1_times;
	std::vector<float> gpu_algo2_times;
	std::vector<float> gpu_algo3_times;

	const int num_repetitions = 1;
	const int start = 30;
	const int end = 31;

	for (int k = start; k < end; k++)
	{
		std::cout<<k<<std::endl;
		const int N = 1<<k;
		auto hX = std::vector<T>(N);

		for (int i = 0; i< N; i++){
			hX[i] = T(1);
		}
		auto dX = gpuVector<T>(hX);

		float s1(0);
		float s2(0);
		float s3(0);

		gpuVector<T> d_output(1);
		thrust::device_ptr<T> dp = thrust::device_pointer_cast(dX.data());

		for (int r = 0; r < num_repetitions; r++)
		{
			//s1 += gputimeit(reduce_kernel<Plus, T>, dX.data(), d_output.data(), dX.size(), ReductionMethod::shared_mem_reduction);
			s2 += gputimeit(reduce_kernel<Plus, T>, dX.data(), d_output.data(), dX.size(), ReductionMethod::warp_reduction);
			//s3 += cputimeit([&](){
			//	thrust::reduce(dp, dp+N);
			//});
		}

		gpu_algo1_times.push_back(s1/num_repetitions);
		gpu_algo2_times.push_back(s2/num_repetitions);
		gpu_algo3_times.push_back(s3/num_repetitions);

	}

	std::cout<<"p = ["<<std::endl;

	for (size_t k = start; k < end; k++)
	{
		const int pos = k-start;
		std::cout<<gpu_algo1_times[pos]<<"\t"<<gpu_algo2_times[pos]<<"\t"<<gpu_algo3_times[pos]<<std::endl;
	}
	std::cout<<"];"<<std::endl;

}

int main()
{
	cudaDeviceProp prop;

	cudaSetDevice(0);

	cudaGetDeviceProperties(&prop, 0);
	std::cout<<prop.name<<std::endl;

	std::cout<<"Using device "<<0<<":"<<prop.name<<std::endl;

	//simple_test_reduction<float>();

	run_reduction_matlab<float>();

    return 0;
}
