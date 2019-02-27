#include "gpuvector.cuh"
#include "unary.cuh"
#include "perf.h"
#include "operators.cuh"

#include <vector>
#include <iostream>
#include <numeric>
using namespace std;

template <typename T>
void simple_test_transform()
{
	std::cout<<"start test"<<std::endl;
	std::vector<T> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,0};

	gpuVector<T> gv(v);

	for(size_t i = 0; i<v.size();i++)
	{
		v[i] = Add<1,T>::apply(v[i]);
	}

	unary_op<Add<1,T>>(gv);
	auto gvh = gv.gather();

	for (size_t i =0; i< v.size(); i++)
	{
		if (gvh[i] != v[i])
		{

			std::cout<<"error:"<<std::endl;
			std::cout<<gv<<std::endl;
			return;
		}
	}

	std::cout<<"done"<<std::endl;
	std::cout<<gv<<std::endl;
}

template<typename T>
void run_transform_matlab()
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

		float s1(0);
		for (int r = 0; r < num_repetitions; r++)
		{
			s1 += gputimeit(unary_op<Add<1,T>, T>, dX);
		}

		gpu_algo1_times.push_back(s1/num_repetitions);
	}

	std::cout<<"p = ["<<std::endl;

	for (size_t k = start; k < end; k++)
	{
		const int pos = k-start;
		std::cout<<gpu_algo1_times[pos]<<std::endl;
	}
	std::cout<<"];"<<std::endl;

}

int main(){
	cudaDeviceProp prop;

	cudaSetDevice(0);

	cudaGetDeviceProperties(&prop, 0);
	std::cout<<prop.name<<std::endl;

	std::cout<<"Using device "<<0<<":"<<prop.name<<std::endl;
	//simple_test_transform<float>();
	run_transform_matlab<float>();

    return 0;
}
