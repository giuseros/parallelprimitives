#include "operators.cuh"
#include "gpuvector.hpp"
#include "scan.cuh"

#include <vector>
#include <iostream>
#include <numeric>

using namespace pp;
using namespace std;

template <typename T>
void simple_test_scan()
{
	std::cout<<"start test"<<std::endl;
	std::vector<T> v{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,0};
	gpuVector<T> gv(v);

	for(size_t i = 1; i<v.size();i++)
	{
		v[i] += v[i-1];
	}


	auto s = scan<Plus, ScanKind::exclusive, T>(gv);
	auto sh = s.gather();

	for (size_t i =0; i< v.size(); i++)
	{
		if (sh[i] != v[i])
		{
			std::cout<<"error:"<<std::endl;
			std::cout<<s<<std::endl;
			return;
		}
	}
	std::cout<<"done"<<std::endl;
}

int main(){
	cudaDeviceProp prop;

	cudaSetDevice(0);

	cudaGetDeviceProperties(&prop, 0);
	std::cout<<prop.name<<std::endl;

	std::cout<<"Using device "<<0<<":"<<prop.name<<std::endl;

	simple_test_scan<float>();

    return 0;
}
