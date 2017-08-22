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
        buffer[threadIdx.x] = 0;
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
    size_t const N(10);
    auto X = createVector<double>(N);
    auto x = createVector<double>(1);
    reduceSum<<<1, 256, 256*sizeof(double)>>>(X, x, N);
    auto y = gatherAndDestroy<double>(x, 1);
    cout<<y<<endl;

    return 0;
}
