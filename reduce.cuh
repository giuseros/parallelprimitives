template<typename T, typename OP>
__global__
void reduceBlock(T *X, T* result, size_t const N)
{
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;

    extern __shared__ T buffer[];
    if (idx < N){
        buffer[threadIdx.x] = X[idx];
    }else{
        buffer[threadIdx.x] = OP::identity();
    }
    __syncthreads();

    for (int offset = blockDim.x/2; offset!=0; offset>>=1){
        if (threadIdx.x < offset){
            buffer[threadIdx.x] = OP::apply(buffer[threadIdx.x]) + buffer[threadIdx.x+offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        *result = buffer[threadIdx.x];
    }
}
