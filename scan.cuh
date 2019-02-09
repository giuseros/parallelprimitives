#include "config.cuh"
#include "operators.cuh"

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

template<typename OP, ScanKind Kind, class T>
__global__ void scan_kernel_kogge_stone(T *ptr, T *block_results, size_t N)
{
	__shared__ T buffer[BLOCK_SIZE];

	local_buffer[THREAD_GRANULARITY];

	// Get the index of the array
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Fill the buffer
	if ( idx < N )
		buffer[threadIdx.x] = ptr[idx];
	else
		buffer[threadIdx.x] = 0;

	for (int i = 0; i< THREAD_GRANULARITY; i++)
		local_buffer[i] = ptr[THREAD_GRANULARITY * idx + i];

	__syncthreads();

	// Get the value in buffer
	T val = scan_block_kogge_stone<OP, Kind>(buffer, threadIdx.x);

	if (block_results != nullptr && threadIdx.x == BLOCK_SIZE-1) {
		block_results[blockIdx.x] = val;
	}

	if (idx < N){
		ptr[idx] = val;
	}
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

template <typename OP, typename T>
__global__ void propagate(T *X, T* tmp, int N)
{
	int bidx = blockIdx.x;
	int idx = threadIdx.x + bidx*BLOCK_SIZE;
	if (idx < N && bidx > 0)
	{
		T acc = tmp[bidx-1];
		X[idx] += acc;
	}
}

template <typename OP, ScanKind kind, typename T>
void scan_block(T * data, T * tmp_data, size_t N, size_t num_blocks, ScanMethod method)
{
	if (method == ScanMethod::brent_kung)
	{
		scan_block_brent_kung<Plus, kind><<<num_blocks, BLOCK_SIZE>>>(data, tmp_data, N);
	}
	else if (method == ScanMethod::kogge_stone)
	{
		scan_kernel_kogge_stone<Plus, kind><<<num_blocks, BLOCK_SIZE>>>(data, tmp_data, N);
	}
}

template <typename OP, ScanKind kind, typename T>
void scan_in_place(gpuVector<T>& X, ScanMethod method = ScanMethod::brent_kung)
{
	int num_blocks = divUp(X.size(), BLOCK_SIZE);

	if (num_blocks == 1){
		scan_block<OP, kind>(X.data(), (T*)nullptr, X.size(), 1, method);
		return;
	}

	gpuVector<T> tmp(num_blocks);
	scan_block<OP, kind>(X.data(), tmp.data(), X.size(), num_blocks, method);
	scan_in_place<OP, kind, T>(tmp);
	propagate<Plus, T><<<num_blocks, BLOCK_SIZE>>>(X.data(), tmp.data(), X.size());
}

template<typename OP, ScanKind kind, typename T>
gpuVector<T> scan(gpuVector<T> const& X, ScanMethod method = ScanMethod::brent_kung)
{
	auto Y = gpuVector<T>(X);
	scan_in_place<OP, kind>(Y, method);
	return Y;
}
