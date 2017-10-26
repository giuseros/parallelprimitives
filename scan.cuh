#include "config.cuh"
#include "operators.cuh"

using flag_type = bool;

// Sequential scan

enum ScanKind{
	inclusive, exclusive
};

template<class OP, ScanKind Kind, class T>
__device__ T scanWarp(volatile T *ptr, const unsigned int idx = threadIdx.x)
{
	const unsigned int lane = idx&31; // index in the warp
	if (lane >= 1) ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]);
	if (lane >= 2) ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]);
	if (lane >= 4) ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]);
	if (lane >= 8) ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]);
	if (lane >= 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

	if (Kind == inclusive)
		return ptr[idx];
	else
		return (lane>0 ? ptr[idx-1] : OP::identity());

}

template<class OP, ScanKind Kind, class T>
__device__ T scanBlock(volatile T *ptr, size_t const idx = threadIdx.x)
{
	const unsigned int lane = idx & 31;
	const unsigned int warpid = idx >> 5;

	T val = scanWarp<OP, Kind>(ptr, idx);
	__syncthreads();

	if (lane==31) ptr[warpid] = ptr[idx];
	__syncthreads();

	if (warpid==0) scanWarp<OP, Kind>(ptr, idx);
	__syncthreads();

	if (warpid >0) val = OP::apply(ptr[warpid-1], val);
	__syncthreads();

	ptr[idx] = val;
	__syncthreads();

	return val;

}

template<class OP, ScanKind Kind, class T>
__global__ void scanModerngpuKernel1(T *ptr, T *block_results, size_t N)
{
	extern __shared__ T buffer[];

	// Get the index of the array
	size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	// Fill the buffer
	if ( idx < N )
		buffer[threadIdx.x] = ptr[idx];
	else
		buffer[threadIdx.x] = 0;
	__syncthreads();

	// Get the value in buffer
	T val = scanBlock<OP, Kind>(buffer, threadIdx.x);

	if (threadIdx.x == blockDim.x-1) {
		block_results[blockIdx.x] = val;
	}

	if (idx < N){
		ptr[idx] = val;
	}
}

template<class OP, ScanKind Kind, class T>
__global__ void scanModerngpuKernel2(T *block_results, size_t numBlocks)
{
	extern __shared__ T buffer[];
	size_t idx = threadIdx.x;
	if (idx < numBlocks){
		buffer[idx] = block_results[idx];
		T val = scanBlock<OP, Kind>(buffer, idx);
		block_results[idx] = val;
	}
}

template<class OP, ScanKind Kind, class T>
__global__ void scanModerngpuKernel3(T *ptr, T *block_results, size_t N)
{
	size_t idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < N && blockIdx.x > 0){
		ptr[idx] = OP::apply(block_results[blockIdx.x -1], ptr[idx]);
	}

}

template <typename OP, ScanKind kind, typename T>
void parallelScanInPlace(gpuVector<T>& X)
{

	size_t const N = X.size();
	size_t numBlocks = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

	gpuVector<T> blockResults(numBlocks);

	// Step 1,2
	scanModerngpuKernel1<OP, kind><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(X.data(), blockResults.data(), N);

	// Step 3
	if (numBlocks > BLOCK_SIZE){
		parallelScanInPlace<OP, kind>(blockResults);
	} else {
		scanModerngpuKernel2<OP, kind><<<1, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(blockResults.data(), numBlocks);
	}

	// Step 4
	scanModerngpuKernel3<OP, kind><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(X.data(), blockResults.data(), N);
}

template<typename OP, ScanKind kind, typename T>
gpuVector<T> parallelScan(gpuVector<T> const& X)
{
	auto Y = gpuVector<T>(X);
	parallelScanInPlace<OP, kind>(Y);
	return Y;
}

template<class OP, ScanKind Kind, class T>
__device__ T segscanWarp(volatile T *ptr, volatile flag_type *hd, const unsigned int idx = threadIdx.x)
{
	const unsigned int lane = idx & 31;

	if (hd[idx]) hd[idx] = lane;

	flag_type minindex = scanWarp<Max, inclusive>(hd);

	if (idx >= minindex + 1) ptr[idx] = OP::apply(ptr[idx-1], ptr[idx]);
	if (idx >= minindex + 2) ptr[idx] = OP::apply(ptr[idx-2], ptr[idx]);
	if (idx >= minindex + 4) ptr[idx] = OP::apply(ptr[idx-4], ptr[idx]);
	if (idx >= minindex + 8) ptr[idx] = OP::apply(ptr[idx-8], ptr[idx]);
	if (idx >= minindex + 16) ptr[idx] = OP::apply(ptr[idx-16], ptr[idx]);

	if (Kind == inclusive)
		return ptr[idx];
	else
		return (lane > 0 && minindex != lane ? ptr[idx-1] : OP::identity());

}

// Segmented scan


template<class OP, ScanKind Kind, class T>
__device__ T segscanBlock(volatile T * ptr, volatile flag_type *hd, const unsigned int idx = threadIdx.x)
{
	unsigned int warpid = idx>>5;
	unsigned int warp_first = warpid<<5;
	unsigned int warp_last = warp_first + 31;

	bool warp_is_open = (hd[warp_first] == 0);
	__syncthreads();

	T val = segscanWarp<OP, Kind>(ptr, hd, idx);

	T warp_total = ptr[warp_last];

	flag_type warp_flag = hd[warp_last]!=0 || !warp_is_open;
	bool will_accumulate = warp_is_open && hd[idx] == 0;

	__syncthreads();
	if (idx == warp_last){
		ptr[warpid] = warp_total;
		hd[warpid] = warp_flag;

	}

	__syncthreads();

	if (warpid == 0)
		segscanWarp<OP, inclusive>(ptr, hd, idx);

	__syncthreads();

	if (warpid != 0 && will_accumulate)
		val = OP::apply(ptr[warpid -1], val);

	__syncthreads();

	ptr[idx] = val;
	__syncthreads();

	return val;
}

template <typename T>
__global__
void segscanKernel(T * X, flag_type * partitions, size_t const N)
{
	extern __shared__ T buffer[];
	extern __shared__ flag_type hd[];

	size_t const idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N){
		buffer[idx] = X[idx];
		hd[idx] = partitions[idx];

		segscanBlock(buffer, hd, idx);

		X[idx] = buffer[idx];
	}
}
