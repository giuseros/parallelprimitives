#include "utils.cuh"
#include "constants.cuh"

enum class ReductionMethod
{
	shared_mem_reduction,
	warp_reduction
};

template <typename OP, typename T>
inline __device__
T reduce_vector_2(const typename vector_t<T,2>::vec_type v)
{
	return OP::apply(v.x, v.y);
}

template <typename OP, typename T>
inline __device__
T reduce_vector_4(const typename vector_t<T,4>::vec_type v)
{
	T t1 = OP::apply(v.x, v.y);
	T t2 = OP::apply(v.z, v.w);
	return OP::apply(t1, t2);
}

template <typename OP, typename T>
inline __device__
T reduce_multiple_elements_2(const T *in, size_t N)
{
	T sum = OP::identity(T());

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N/2; i += blockDim.x * gridDim.x)
	{
		auto val = reinterpret_cast<const typename vector_t<T,2>::vec_type *>(in)[i];

		sum = OP::apply(sum, reduce_vector_2<OP,T>(val));

	}
	return sum;
}

template <typename OP, typename T>
inline __device__
T reduce_multiple_elements_4(const T *in, size_t N)
{
	T sum = OP::identity(T());

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N/4; i += blockDim.x * gridDim.x)
	{
		auto val = reinterpret_cast<const typename vector_t<T,4>::vec_type *>(in)[i];

		sum = OP::apply(sum, reduce_vector_4<OP,T>(val));

	}
	return sum;
}

// Warp reduction
template<typename OP, unsigned int blockSize, typename T>
__device__ void warp_reduce(volatile T *buffer, int tidx)
{
	if (blockSize >= 64)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 32]);
	}
	if (blockSize >= 32)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 16]);
	}
	if (blockSize >= 16)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 8]);
	}
	if (blockSize >= 8)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 4]);
	}
	if (blockSize >= 4)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 2]);
	}
	if (blockSize >= 2)
	{
		buffer[tidx] = OP::apply(buffer[tidx], buffer[tidx + 1]);
	}
}

/**
 * This code has been taken from https://moderngpu.github.io/scan.html,
 * but I am not able to make it work
 */
__device__ int shfl_add(int x, int offset, int width = 32) {
	int result = 0;
#if __CUDA_ARCH__ >= 300
	int mask = (32 - width)<< 8;
	asm(
			"{.reg .s32 r0;"
			".reg .pred p;"
			"shfl.up.b32 r0|p, %1, %2, %3;"
			"@p add.s32 r0, r0, %4;"
			"mov.s32 %0, r0; }"
			: "=r"(result) : "r"(x), "r"(offset), "r"(mask), "r"(x));
#endif
	return result;
}

#define FULL_MASK 0xffffffff

template <typename OP, typename T>
__inline__ __device__
T warp_reduce_shfl(T val)
{

#pragma unroll
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val = OP::apply(val, __shfl_down_sync(FULL_MASK, val, offset));
	}

//#pragma unroll
//for (int offset = 1; offset < 32; offset*=2)
//{
//	val = shfl_add(val, offset);
//}
	return val;
}

template <typename OP, typename T>
__inline__ __device__
T block_reduce_sum(T val) {

	static __shared__ T shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warp_reduce_shfl<OP>(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	// read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warp_reduce_shfl<OP>(val); //Final reduce within first warp

	return val;
}

template<typename OP, unsigned int blockSize, typename T>
__global__
void device_reduce_kernel_1(const T *X, T* out, size_t const N)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int tidx = threadIdx.x;

	T sum = reduce_multiple_elements_4<OP>(X, N);

	static __shared__ T shared_buffer[BLOCK_SIZE];

	if (idx < N)
	{
		shared_buffer[tidx] = sum;
	}
	else
	{
		shared_buffer[tidx] = T(0);
	}
	__syncthreads();

	if (blockSize >= 1024){
		if (tidx < 512)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx+512]);
		}
		__syncthreads();
	}

	if (blockSize >= 512){
		if (tidx < 256)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx+256]);
		}
		__syncthreads();
	}

	if (blockSize >= 256){
		if (tidx < 128)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx+128]);
		}
		__syncthreads();
	}

	if (blockSize >= 128){
		if (tidx < 64)
		{
			shared_buffer[tidx] = OP::apply(shared_buffer[tidx], shared_buffer[tidx+64]);
		}
		__syncthreads();
	}

	if (tidx < 32)
	{
		warp_reduce<OP, blockSize>(shared_buffer, tidx);
	}

	if (threadIdx.x == 0)
	{
		OP::apply_atomic(out, shared_buffer[0]);
	}
}

template <typename OP, typename T>
__global__ void device_reduce_kernel_2(const T *in, T* out, size_t N) {
	//reduce multiple elements per thread

	T sum = reduce_multiple_elements_4<OP>(in, N);

	sum = block_reduce_sum<OP>(sum);

	if (threadIdx.x == 0)
	{
		OP::apply_atomic(out, sum);
	}
	// I tested with warp_reduce+atomic, but it was actually slower
	//	sum = warp_reduce_shfl<OP>(sum);
	//
	//	if ((threadIdx.x & (warpSize - 1)) == 0)
	//	{
	//		OP::apply_atomic(out, sum);
	//	}
}

template<typename OP, typename T>
void reduce_kernel(const T * d_in, T * d_out, size_t N, ReductionMethod method)
{
	const int num_blocks = min(divUp(N, BLOCK_SIZE), size_t(MAX_BLOCKS));
//	const int num_blocks = divUp(N, BLOCK_SIZE);
	switch(method)
	{
	case ReductionMethod::shared_mem_reduction:
		device_reduce_kernel_1<OP, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
		break;
	case ReductionMethod::warp_reduction:
		device_reduce_kernel_2<OP><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
		break;
	default:
		break;
	}
}


template<typename OP, typename T>
T reduce(const gpuVector<T>& v, ReductionMethod method = ReductionMethod::warp_reduction)
{
	gpuVector<T> d_output(1);

	reduce_kernel<OP>(v.data(), d_output.data(), v.size(), method);

	T output(0);
	cudaMemcpy(&output, d_output.data(), sizeof(T), cudaMemcpyDeviceToHost);

	return T(output);
}
