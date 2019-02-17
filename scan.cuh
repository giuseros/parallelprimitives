#ifndef SCAN_CUH__
#define SCAN_CUH__

#include "config.cuh"
#include "operators.cuh"
#include "reduce.cuh"

using flag_type = bool;

// Sequential scan

enum class ScanKind{
	inclusive, exclusive
};

enum class ScanMethod
{
	brent_kung, kogge_stone
};

template<typename OP, ScanKind Kind, class T>
__device__ T scan_block_brent_kung(volatile T *block_results, size_t const idx = threadIdx.x)
{

	// Upsweep
	for (size_t stride = 1; stride < blockDim.x; stride *=2)
	{
		__syncthreads();
		int index = (idx+1) * 2 * stride -1;
		if (index < blockDim.x)
		{
			block_results[index] += block_results[index-stride];
		}
	}

	// Downsweep
	for (size_t stride = BLOCK_SIZE/4; stride > 0; stride /= 2)
	{
		__syncthreads();
		int index = (idx+1) * 2 * stride -1;
		if (index + stride  < BLOCK_SIZE)
		{
			block_results[index+stride] += block_results[index];
		}
	}
	__syncthreads();

	return block_results[idx];
}

template <typename OP, ScanKind kind, typename T>
__global__ void scan_block_brent_kung(T *ptr, T *intermediate_results, int N)
{
	__shared__ T block_results[BLOCK_SIZE];
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx< N)
	{
		block_results[threadIdx.x] = ptr[idx];
	}
	else
	{
		block_results[threadIdx.x] = 0;
	}

	T val = scan_block_brent_kung<OP, kind>(block_results, threadIdx.x);

	ptr[idx] = val;

	if (intermediate_results != nullptr && threadIdx.x == BLOCK_SIZE -1)
	{
		intermediate_results[blockIdx.x] = block_results[threadIdx.x];
	}
}

template<typename OP, ScanKind Kind, class T>
__device__ T scan_warp_kogge_stone(volatile T *ptr, const unsigned int idx = threadIdx.x)
{
	// Kogge-stone warp scan
	const unsigned int lane = idx & 31; // index in the warp ( idx & 0..011111 )

	// Those operations are synchronised, so no need of any double buffering
	if (lane >= 1) ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]);
	if (lane >= 2) ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]);
	if (lane >= 4) ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]);
	if (lane >= 8) ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]);
	if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

	if (Kind == ScanKind::inclusive)
		return ptr[idx];
	else
		return (lane>0 ? ptr[idx-1] : OP::identity(T()));

}

template<typename OP, ScanKind Kind, class T>
__device__ T scan_block_kogge_stone(volatile T *ptr, size_t const idx = threadIdx.x)
{
	const unsigned int lane = idx & 31;
	const unsigned int warpid = idx >> 5;

	T val = scan_warp_kogge_stone<OP, Kind>(ptr, idx);
	__syncthreads();

	if (lane==31) ptr[warpid] = ptr[idx];
	__syncthreads();

	if (warpid==0) scan_warp_kogge_stone<OP, Kind>(ptr, idx);
	__syncthreads();

	if (warpid >0) val = OP::apply(ptr[warpid-1], val);
	__syncthreads();

	ptr[idx] = val;
	__syncthreads();

	return val;

}

template<typename OP, ScanKind Kind, int granularity, class T>
__global__ void scan_kernel_kogge_stone(T *ptr, T *block_results, size_t N)
{
	extern __shared__ T buffer[];

	T local_buffer[granularity];

	// Get the index of the array
	size_t idx = granularity*(threadIdx.x + blockDim.x * blockIdx.x);
	T start = 0;

	if (block_results != nullptr)
	{
		start = block_results[blockIdx.x];
	}
#pragma unroll
	for (int i = 0; i< granularity && idx + i < N; i++)
	{
		local_buffer[i] = ptr[idx + i];
	}

	T local_reduction = 0;

#pragma unroll
	for (int i = 0; i< granularity; i++)
	{
		local_reduction += local_buffer[i];
	}

	buffer[threadIdx.x] = local_reduction;

	__syncthreads();

	// Get the value in buffer
	T val = scan_block_kogge_stone<OP, ScanKind::exclusive>(buffer, threadIdx.x);

	int i_start = 0;

	if (Kind == ScanKind::exclusive && idx < N){
		ptr[idx] = (threadIdx.x  >=1 ? buffer[threadIdx.x -1] : 0);
		i_start = 1;
	}

#pragma unroll
	for (int i = 1; i < granularity; i++)
	{
		local_buffer[i] += local_buffer[i-1];
	}

	if (granularity > 1)
	{
#pragma unroll
		for (int i = i_start; i < granularity && idx + i < N; i++)
		{
			ptr[idx + i] = local_buffer[i - i_start] + val + start;
		}
	}
	else
	{
		ptr[idx + i_start] = local_buffer[0] + val + start;
	}
}

template <typename OP, ScanKind kind, typename T>
void scan_in_place(gpuVector<T>& X)
{
	if (X.size() <= 256)
	{
		scan_kernel_kogge_stone<OP, kind, 1><<<1, 256, 256*sizeof(T)>>>(X.data(), (T*)nullptr, X.size());
	}
	else if (X.size() <= 3*256)
	{
		scan_kernel_kogge_stone<OP, kind, 3><<<1, 256, 256*sizeof(T)>>>(X.data(), (T*)nullptr, X.size());
	}
	else if(X.size() <= 5*512)
	{
		scan_kernel_kogge_stone<OP, kind, 5><<<1, 512, 512*sizeof(T)>>>(X.data(), (T*)nullptr, X.size());
	}
	else
	{
		int num_blocks = divUp(X.size(), 128);
		gpuVector<T> tmp(num_blocks);

		device_reduce_kernel_3<OP><<<num_blocks, 8, 8*sizeof(T)>>>(X.data(), tmp.data(), X.size());
		scan_in_place<OP, ScanKind::exclusive, T>(tmp);
		scan_kernel_kogge_stone<OP, kind, 7><<<1, 128, 128*sizeof(T)>>>(X.data(), tmp.data(), X.size());
	}
}

template<typename OP, ScanKind kind, typename T>
gpuVector<T> scan(gpuVector<T> const& X)
{
	auto Y = gpuVector<T>(X);
	scan_in_place<OP, kind>(Y);
	return Y;
}

#endif // SCAN_CUH__
