#include "gpuvector.cuh"
#include "utils.cuh"
#include "perf.h"
#include "scan.cuh"

#include <thrust/scan.h>

#include <vector>
#include <iostream>
#include <numeric>
using namespace std;

template <typename T>
void simple_test_scan()
{
	gpuVector<float> v ({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15});
	std::cout<<scan<Plus, ScanKind::inclusive>(v, ScanMethod::brent_kung)<<std::endl;
}

template<typename T>
void run_scan_matlab()
{
	std::vector<float> gpu_algo1_times;
	std::vector<float> gpu_algo3_times;

	const int num_repetitions = 10;
	const int start = 10;
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

		float s1(0), s3(0);

		gpuVector<T> d_output(1);
		thrust::device_ptr<T> dp = thrust::device_pointer_cast(dX.data());

		for (int r = 0; r < num_repetitions; r++)
		{
			//s1 += gputimeit(reduce_kernel<Plus, T>, dX.data(), d_output.data(), dX.size(), ReductionMethod::shared_mem_reduction);
			s1 += gputimeit(scan_in_place<Plus, ScanKind::inclusive,T>, dX, ScanMethod::kogge_stone);
//			s3 += cputimeit([&](){
//				thrust::inclusive_scan(dp, dp+N, dp);
//			});
		}

		gpu_algo1_times.push_back(s1/num_repetitions);
//		gpu_algo2_times.push_back(s2/num_repetitions);
		gpu_algo3_times.push_back(s3/num_repetitions);

	}

	std::cout<<"p = ["<<std::endl;

	for (size_t k = start; k < end; k++)
	{
		const int pos = k-start;
		std::cout<<gpu_algo1_times[pos]<<std::endl;
//		std::cout<<gpu_algo1_times[pos]<<std::endl;
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

	run_scan_matlab<float>();

    return 0;
}
