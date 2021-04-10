#include "operators.cuh"
#include "gpuvector.hpp"
#include "histogram.cuh"

#include <vector>
#include <iostream>
#include <numeric>

using namespace pp;
using namespace std;

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> const &hVec)
{
	for (auto e : hVec)
	{
		out << e << " ";
	}
	return out;
}

template <typename T>
void simple_test_histogram()
{
	std::cout<<"start test"<<std::endl;
	std::vector<T> v{1,1,1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1};
    std::vector<T> hr(3);
	gpuVector<T> gv(v);

	for(size_t i = 0; i<v.size();i++){
		hr[v[i]]++;
	}

	auto hd = pp::histogram(gv, 2);
	auto hh = hd.gather();

	for (size_t i =0; i< hr.size(); i++)
	{
		if (hh[i] != hr[i])
		{
			std::cout<<"error:"<<std::endl;
			std::cout<<hh<<std::endl;
			std::cout<<hr<<std::endl;
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

	simple_test_histogram<int>();

    return 0;
}
