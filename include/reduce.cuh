/**
 * Copyright (c) 2019 Giuseppe ROssini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef REDUCE_HH__
#define REDUCE_HH__
#include "operators.cuh"
#include "traits.cuh"
#include "gpuvector.hpp"

#define REDUCE_BLOCK_SIZE 256
#define REDUCE_MAX_BLOCKS size_t(1024)
#define REDUCE_WARP_SHARED (32)
#define REDUCE_FULL_MASK 0xffffffff

#define divUp(x, y) ((x + y - 1) / (y))

using namespace pp;

// Default max number of blocks for reduction

enum class ReductionMethod
{
	shared_mem_reduction,
	warp_reduction
};

template <typename OP, typename T, int K>
inline __device__
	T
	reduce_vector(const typename vector_t<T, K>::vec_type v)
{
	if (K == 2)
	{
		return OP::apply(v.x, v.y);
	}
	else
	{
		T t1 = OP::apply(v.x, v.y);
		T t2 = OP::apply(v.z, v.w);
		return OP::apply(t1, t2);
	}
}

template <typename OP, typename T, int K>
inline __device__
	T
	reduce_multiple_elements(const T *in, size_t N)
{
	T sum = OP::identity(T());

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N / K; i += blockDim.x * gridDim.x)
	{
		auto val = reinterpret_cast<const typename vector_t<T, K>::vec_type *>(in)[i];

		sum = OP::apply(sum, reduce_vector<OP, T, K>(val));
	}
	return sum;
}

template <typename OP, typename T>
__inline__ __device__
	T
	warp_reduce_shfl(T val)
{

#pragma unroll
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		val = OP::apply(val, __shfl_down_sync(REDUCE_FULL_MASK, val, offset));
	}

	return val;
}

template <typename OP, typename T>
__inline__ __device__
	T
	block_reduce_sum(T val)
{

	static __shared__ T shared[REDUCE_WARP_SHARED]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warp_reduce_shfl<OP>(val); // Each warp performs partial reduction

	if (lane == 0)
		shared[wid] = val; // Write reduced value to shared memory

	__syncthreads(); // Wait for all partial reductions

	// read from shared memory only if that warp existed
	if (wid == 0)
		val = warp_reduce_shfl<OP>(shared[lane]); //Final reduce within first warp

	return val;
}

template <typename OP, unsigned int blockSize, typename T, bool reduce_scan = false>
__global__ void device_reduce_shared_mem(const T *X, T *out, size_t const N)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidx = threadIdx.x;

	T sum = reduce_multiple_elements<OP, T, 4>(X, N);

	static __shared__ volatile T shared_buffer[REDUCE_BLOCK_SIZE];

	if (idx < N)
	{
		shared_buffer[tidx] = sum;
	}
	else
	{
		shared_buffer[tidx] = T(0);
	}
	__syncthreads();

	if (blockSize >= 1024)
	{
		if (tidx < 512)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 512]);
		}
		__syncthreads();
	}

	if (blockSize >= 512)
	{
		if (tidx < 256)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 256]);
		}
		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tidx < 128)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 128]);
		}
		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tidx < 64)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 64]);
		}
		__syncthreads();
	}

	if (tidx < 32)
	{
		// warp_reduce<OP, blockSize>(shared_buffer, tidx);
		if (blockSize >= 64)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 32]);
		}
		if (blockSize >= 32)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 16]);
		}
		if (blockSize >= 16)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 8]);
		}
		if (blockSize >= 8)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 4]);
		}
		if (blockSize >= 4)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 2]);
		}
		if (blockSize >= 2)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx + 1]);
		}
	}

	if (tidx == 0)
	{
		if (reduce_scan){
			out[blockIdx.x] = shared_buffer[threadIdx.x];
		} else {
			// We need to atomically apply the operation to the output
			OP::apply_atomic(out, shared_buffer[0]);
		}
		
	}
}

template <typename OP, typename T>
__global__ void device_reduce_warp(const T *in, T *out, size_t N)
{
	//reduce multiple elements per thread
	T sum = reduce_multiple_elements<OP, T, 4>(in, N);

	sum = block_reduce_sum<OP>(sum);

	if (threadIdx.x == 0)
	{
		OP::apply_atomic(out, sum);
	}
}

template <typename OP, typename T>
void reduce_kernel(const T *d_in, T *d_out, size_t N, ReductionMethod method)
{
	const int num_blocks = min(divUp(N, REDUCE_BLOCK_SIZE), size_t(REDUCE_MAX_BLOCKS));
	switch (method)
	{
	case ReductionMethod::shared_mem_reduction:
		device_reduce_shared_mem<OP, REDUCE_BLOCK_SIZE><<<num_blocks, REDUCE_BLOCK_SIZE>>>(d_in, d_out, N);
		break;
	case ReductionMethod::warp_reduction:
		device_reduce_warp<OP><<<num_blocks, REDUCE_BLOCK_SIZE>>>(d_in, d_out, N);
		break;
	default:
		break;
	}
}

namespace pp
{

	template <typename OP, typename T>
	T reduce(const gpuVector<T> &v, ReductionMethod method = ReductionMethod::shared_mem_reduction)
	{
		gpuVector<T> d_output(1,0);
		gpuVector<T> v_copy(v);
		
		reduce_kernel<OP>(v_copy.data(), d_output.data(), v.size(), method);

		T output(0);
		cudaMemcpy(&output, d_output.data(), sizeof(T), cudaMemcpyDeviceToHost);
		std::cout << d_output << std::endl;

		return T(output);
	}

}

#endif
