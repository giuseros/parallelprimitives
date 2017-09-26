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

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

template <typename T>
__global__
void scanSumNoConflicts(T* X, size_t const n)
{
    extern __shared__ T buffer[];

    size_t const idx = threadIdx.x;
    size_t offset = 1;

    buffer[2*idx] = X[2*idx];
    buffer[2*idx+1] = X[2*idx+1];

    // Down-sweep
    for (size_t d = (n >> 1); d > 0; d >>= 1){
        __syncthreads();

        if (idx < d){
            size_t const element1 = offset*(2*idx+1)-1;
            size_t const element2 = offset*(2*idx+2)-1;
            buffer[element2] += buffer[element1];
        }
        offset *= 2;
    }

    // Up-sweep
    if (idx == 0){ buffer[n-1] = 0; }

    for (size_t d = 1; d < n; d *= 2){

        offset >>= 1;
        __syncthreads();

        if (idx < d){
            size_t const element1 = offset*(2*idx+1)-1;
            size_t const element2 = offset*(2*idx+2)-1;

            T tmp = buffer[element1];
            buffer[element1] = buffer[element2];
            buffer[element2] += tmp;
        }

    }

    __syncthreads();
    X[2*idx] = buffer[2*idx];
    X[2*idx+1] = buffer[2*idx+1];
}

template <typename T>
__global__
void scanSum(T* X, size_t const n)
{
    extern __shared__ T buffer[];

    size_t const idx = threadIdx.x;
    size_t offset = 1;

    buffer[2*idx] = X[2*idx];
    buffer[2*idx+1] = X[2*idx+1];

    // Down-sweep
    for (size_t d = (n >> 1); d > 0; d >>= 1){
        __syncthreads();

        if (idx < d){
            size_t const element1 = offset*(2*idx+1)-1;
            size_t const element2 = offset*(2*idx+2)-1;
            buffer[element2] += buffer[element1];
        }
        offset *= 2;
    }

    // Up-sweep
    if (idx == 0){ buffer[n-1] = 0; }

    for (size_t d = 1; d < n; d *= 2){

        offset >>= 1;
        __syncthreads();

        if (idx < d){
            size_t const element1 = offset*(2*idx+1)-1;
            size_t const element2 = offset*(2*idx+2)-1;

            T tmp = buffer[element1];
            buffer[element1] = buffer[element2];
            buffer[element2] += tmp;
        }

    }

    __syncthreads();
    X[2*idx] = buffer[2*idx];
    X[2*idx+1] = buffer[2*idx+1];
}


template <typename T>
T *createVector(size_t N)
{
    std::vector<T> X(N);
    for (int i = 0; i < N; i++) { X[i] = T(i); }
    cout<<X<<endl;

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
    size_t const N(512);
    auto X = createVector<float>(N);
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    scanSum<<<1, N/2, N*sizeof(float)>>>(X, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    auto Y = gatherAndDestroy<float>(X, N);
    cout<<Y<<endl;
    cout<<time<<endl;

    return 0;
}
