#ifndef HISTOGRAM_HH__
#define HISTOGRAM_HH__

#include <cuda.h>
#include <vector>
#include <iostream>

#define divUp(x, y) ((x + y - 1) / (y))
#define HIST_BLOCK_SIZE 256
#define HIST_MAX_NUM_BLOCKS (size_t(1024))
#define HIST_SHARED_SIZE 256

namespace pp{
    
    namespace kernel{
       
        __global__ void histogram(const int* in, int* out, int N, int max){
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            int stride = blockDim.x * gridDim.x;
            extern __shared__ int buffer[];

            while(idx<N){
                const int v = in[idx];    
                atomicAdd(&buffer[v], 1);
                //buffer[0] = 1;
                idx+= stride;
            }
            __syncthreads();

            if (threadIdx.x <= max){
                atomicAdd(&out[threadIdx.x], buffer[threadIdx.x]);
            }
            
        }
    }


	gpuVector<int> histogram(const gpuVector<int> &v, int max)
	{
		gpuVector<int> d_output(max+1,0);
        const int num_blocks = std::min(divUp(v.size(), HIST_BLOCK_SIZE), HIST_MAX_NUM_BLOCKS);
		std::cout<<num_blocks<<std::endl;
		kernel::histogram<<<num_blocks, HIST_BLOCK_SIZE, (max+1)*sizeof(int)>>>(v.data(), d_output.data(), v.size(), max);

		return d_output;
	}
} // namespace pp

#endif // HISTOGRAM_HH__