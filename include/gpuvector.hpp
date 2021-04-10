#ifndef GPUVECTOR__HH__
#define GPUVECTOR__HH__

#include <cuda.h>
#include <vector>
#include <iostream>

namespace pp{

template <typename T>
struct gpuVector
{
public:
	gpuVector(): fSize(0), fDevPtr(nullptr){ }

	gpuVector(gpuVector<T> const& other): fSize(other.fSize){
		cudaMalloc(&fDevPtr, other.fSize * sizeof(T));
		cudaMemcpy(fDevPtr, other.fDevPtr, other.fSize * sizeof(T), cudaMemcpyDeviceToDevice);
	}

	gpuVector(size_t size): fSize(size)
	{
		cudaMalloc(&fDevPtr, fSize * sizeof(T));
	}

	gpuVector(size_t size, T val) : fSize(size)
	{
		auto valVec = std::vector<T>(size, val);
		cudaMalloc(&fDevPtr, size * sizeof(T));
		cudaMemcpy(fDevPtr, valVec.data(), size*sizeof(T), cudaMemcpyHostToDevice);
	}

	gpuVector(std::vector<T>const& vec): fSize(vec.size())
	{
		cudaMalloc(&fDevPtr, vec.size()*sizeof(T));
		cudaMemcpy(fDevPtr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice);
	}

	std::vector<T> gather() const
	{
		std::vector<T> hostVector(fSize);
		cudaMemcpy(hostVector.data(), fDevPtr, fSize*sizeof(T), cudaMemcpyDeviceToHost);
		return hostVector;
	}

	T * data() { return fDevPtr; }
	T * data() const { return fDevPtr; }
	size_t size() const { return fSize; }

	~gpuVector()
	{
		cudaFree(fDevPtr);
	}

private:
	T *fDevPtr;
	size_t fSize;
};

template <typename T>
std::ostream &operator<<(std::ostream &out, gpuVector<T> const &gpuVec)
{
	auto hVec = gpuVec.gather();
	for (auto e : hVec)
	{
		out << e << " ";
	}
	return out;
}

}
#endif
