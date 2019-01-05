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
	const int size_limit = 1e6;

	for (int N = 0; N < size_limit; N += 100000)
	{
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
			std::cerr<<"Warp reduction failed with N "<<N<<std::endl;
			return;
		}
		if (rgpu2 != rcpu)
		{
			std::cerr<<"Shared mem reduction failed with N "<<N<<std::endl;
			return;
		}
	}
	std::cout<<"Tests passed"<<std::endl;
}

template<typename T>
void run_reduction_matlab()
{
	std::vector<float> gpu_algo1_times;
	std::vector<float> gpu_algo2_times;

	const int num_repetitions = 20;
	const int start = 20;
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

		for (int r = 0; r < num_repetitions; r++)
		{
			s1 += gputimeit(reduce<Plus, T>, dX, ReductionMethod::shared_mem_reduction);
			s2 += gputimeit(reduce<Plus, T>, dX, ReductionMethod::warp_reduction);
		}

		gpu_algo1_times.push_back(s1/num_repetitions);
		gpu_algo2_times.push_back(s2/num_repetitions);

	}

	std::cout<<"p = ["<<std::endl;

	for (size_t k = start; k < end; k++)
	{
		const int pos = k-start;
		std::cout<<gpu_algo1_times[pos]<<"\t"<<gpu_algo2_times[pos]<<std::endl;
	}
	std::cout<<"];"<<std::endl;

}

int main()
{
	simple_test_reduction<int>();

	run_reduction_matlab<int>();

    return 0;
}
