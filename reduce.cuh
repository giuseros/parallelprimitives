#include "utils.cuh"
#include "constants.cuh"

template <typename T, int vec_size>
struct vector_t;

template <>
struct vector_t<double, 2>
{
	using vec_type = double2;
};

template <>
struct vector_t<double, 4>
{
	using vec_type = double4;
};

template <>
struct vector_t<float, 2>
{
	using vec_type = float2;
};

template <>
struct vector_t<float, 4>
{
	using vec_type = float4;
};

template <>
struct vector_t<int, 2>
{
	using vec_type = int2;
};

template <>
struct vector_t<int, 4>
{
	using vec_type = int4;
};

enum class ReductionMethod
{
	shared_mem_reduction,
	warp_reduction
};

template <typename OP, int vec_size, typename T>
inline __device__
T reduce_multiple_elements(T *in, size_t N)
{
	T sum = OP::identity(T());

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N/vec_size; i += blockDim.x * gridDim.x)
	{
		auto val = reinterpret_cast<typename vector_t<T,vec_size>::vec_type *>(in)[i];
		T sum1 = OP::apply(val.x, val.y);
		sum = OP::apply(sum1, sum);
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


template <typename OP, typename T>
__inline__ __device__
T warp_reduce_shfl(T val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
    val = OP::apply(val, __shfl_down(val, offset));
  }
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

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warp_reduce_shfl<OP>(val); //Final reduce within first warp

  return val;
}

template<typename OP, unsigned int blockSize, typename T>
__global__
void device_reduce_kernel_1(T *X, T* out, size_t const N)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int tidx = threadIdx.x;

    T sum = reduce_multiple_elements<OP, 4>(X, N);

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
__global__ void device_reduce_kernel_2(T *in, T* out, size_t N) {
	//reduce multiple elements per thread

	T sum = reduce_multiple_elements<OP, 4>(in, N);

	sum = block_reduce_sum<OP>(sum);

	if (threadIdx.x == 0)
	{
		OP::apply_atomic(out, sum);
	}
//	sum = warp_reduce_shfl<OP>(sum);
//
//	if ((threadIdx.x & (warpSize - 1)) == 0)
//	{
//		OP::apply_atomic(out, sum);
//	}
}


template<typename OP, typename T>
T reduce(const gpuVector<T>& v, ReductionMethod method = ReductionMethod::warp_reduction)
{
	const int num_blocks = min(divUp(v.size(), BLOCK_SIZE), size_t(BLOCK_SIZE));

	gpuVector<T> d_output(1);

	switch(method)
	{
	case ReductionMethod::shared_mem_reduction:
		device_reduce_kernel_1<OP, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(v.data(), d_output.data(), v.size());
		break;
	case ReductionMethod::warp_reduction:
		device_reduce_kernel_2<OP><<<num_blocks, BLOCK_SIZE>>>(v.data(), d_output.data(), v.size());
		break;
	default:
		break;
	}

	T output(0);
	cudaMemcpy(&output, d_output.data(), sizeof(T), cudaMemcpyDeviceToHost);

	return T(output);
}
