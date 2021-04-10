/**
 * Copyright (c) 2019 Giuseppe Rossini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef SCAN_CUH__
#define SCAN_CUH__

#include "operators.cuh"
#include "reduce.cuh"

#define SCAN_BLOCK_SIZE 256

using namespace pp;

enum class ScanKind{
	inclusive, exclusive
};


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

    // Scan each warp and save your local value
	T val = scan_warp_kogge_stone<OP, Kind>(ptr, idx);
	__syncthreads();

    // Store the last lane of the warp in the first part of shared memory
	if (lane==31) ptr[warpid] = ptr[idx];
	__syncthreads();

    // Scan the first warp
	if (warpid==0) scan_warp_kogge_stone<OP, Kind>(ptr, idx);
	__syncthreads();

    // Distribute the scans
	if (warpid >0) val = OP::apply(ptr[warpid-1], val);
	__syncthreads();

    // Store the correct value
	ptr[idx] = val;
	__syncthreads();

	return val;
}

template<typename OP, ScanKind Kind, int granularity, class T>
__global__ void scan_kernel_kogge_stone(T *ptr, T *block_results, size_t N)
{
	extern __shared__ T buffer[];

    /*
     * Perform local reduction and store in buffer
     */
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

    // Start the block scan

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
	// else if (X.size() <= 3*256)
	// {
	// 	scan_kernel_kogge_stone<OP, kind, 3><<<1, 256, 256*sizeof(T)>>>(X.data(), (T*)nullptr, X.size());
	// }
	// else if(X.size() <= 5*512)
	// {
	// 	scan_kernel_kogge_stone<OP, kind, 5><<<1, 512, 512*sizeof(T)>>>(X.data(), (T*)nullptr, X.size());
	// }
	else
	{
        const int reduce_scan_block_size = 128;
        const int num_blocks = min(divUp(X.size(), reduce_scan_block_size), size_t(REDUCE_MAX_BLOCKS));
        const int granularity = 7;
		const bool do_reduce_for_scan = true;

        gpuVector<T> tmp(num_blocks);
		device_reduce_shared_mem<OP, reduce_scan_block_size, T, do_reduce_for_scan><<<num_blocks, reduce_scan_block_size>>>(X.data(), tmp.data(), X.size());
        scan_in_place<OP, ScanKind::exclusive, T>(tmp);
		scan_kernel_kogge_stone<OP, kind, granularity><<<1, reduce_scan_block_size, reduce_scan_block_size*sizeof(T)>>>(X.data(), tmp.data(), X.size());
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
