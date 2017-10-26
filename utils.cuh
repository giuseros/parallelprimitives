#ifndef UTILS_CUH__
#define UTILS_CUH__

#include "gpuvector.cuh"
#include <iostream>

#define divUp(x,y) ((x+y+1)/y)


template<typename T>
std::ostream& operator<<(std::ostream& out,  gpuVector<T> const& gpuVec)
{
	auto hVec = gpuVec.gather();
	for (auto e : hVec){ out<<e<<" ";}
	return out;
}
#endif
