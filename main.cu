#include <iostream>
#include <cuda.h>
#include <vector>

using namespace std;

template<typename T>
__global__
void addOne(T *X, size_t const N){
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx < N) X[idx]++;
}

template<typename T>
__global__
void reduceSum(T *X, T* result, size_t const N)
{
    size_t idx = threadIdx.x + blockDim.x*blockIdx.x;

    extern __shared__ T buffer[];
    if (idx < N){
        buffer[threadIdx.x] = X[idx];
    }else{
        buffer[threadIdx.x] = T(0);
    }
    __syncthreads();

    for (int offset = blockDim.x/2; offset!=0; offset>>=1){
        if (threadIdx.x < offset){
            buffer[threadIdx.x]+=buffer[threadIdx.x+offset];
        }
        __syncthreads();
    }


    if (threadIdx.x == 0){
        *result = buffer[threadIdx.x];
    }
}

template <typename T>
__global__
void scanSum(T* X, size_t const N)
{
    size_t const idx = threadIdx.x;
    extern __shared__ T buffer[];
    if (idx < N){
        buffer[threadIdx.x] = X[idx];
    }else {
        buffer[threadIdx.x] = T(0);
    }

    __syncthreads();

    // Down-sweep
    for (size_t offset = 2; offset <=  blockDim.x; offset *= 2 ){
        if (idx < blockDim.x/offset){
            size_t const element1 = (offset/2-1) + offset*idx;
            size_t const element2 = element1 + offset/2;
            buffer[element2] += buffer[element1];
        }
        __syncthreads();
    }

    // Up-sweep
    buffer[blockDim.x-1] = 0;

    __syncthreads();
    for (size_t offset = blockDim.x; offset >=2; offset /= 2){
        if (idx < blockDim.x/offset){
            size_t const element1 = (offset/2-1) + offset*idx;
            size_t const element2 = element1 + offset/2;

            T tmp = buffer[element2];
            buffer[element2] += buffer[element1];
            buffer[element1] = tmp;
        }
    }


    if (idx < N){
        X[idx] = buffer[idx];
    }

}


template <typename T>
T *createVector(size_t N)
{
    std::vector<T> X(N);
    for (int i = 0; i < N; i++) { X[i] = T(i); }

    T * result;
    cudaMalloc(&result, N*sizeof(T));
    cudaMemcpy(result, X.data(), N*sizeof(T), cudaMemcpyHostToDevice);
    return result;
}

template<typename T>
std::vector<T> gatherAndDestroy(T* X, size_t const N)
{
    auto result = std::vector<T>(N);
    cudaMemcpy(result.data(), X, N*sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(X);
    return result;

}


template<typename T>
std::ostream& operator<<(std::ostream& out, std::vector<T> const& X){
    for (auto const i : X) { out << i << " "; }
    out<<std::endl;
    return out;
}

int main()
{
    size_t const N(8);
    auto X = createVector<double>(N);
    scanSum<<<1, 256, 256*sizeof(double)>>>(X, N);
    auto Y = gatherAndDestroy<double>(X, N);
    cout<<Y<<endl;

    return 0;
}
