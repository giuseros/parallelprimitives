#ifndef BULK_CUH__
#define BULK_CUH__

#include "utils.cuh"
#include "config.cuh"


template <typename T, typename inttype>
__global__
void bulkRemoveKernel(T * output,
		T const *  input, size_t sizeInput,
		inttype const * partitions,
		inttype const * gindices, size_t sizeIndices)
{
	extern __shared__ inttype sindices[];

	size_t idx = threadIdx.x;

	inttype p0 = partitions[blockIdx.x];
	inttype p1 = partitions[blockIdx.x+1];

	sindices[threadIdx.x] = (idx < sizeInput);

	__syncthreads();

	// How many indices to remove?
	inttype indexCount = p1-p0; // number of indices to remove

	// Only indexCount threads will update the shared memory.
	if (idx < indexCount){
		inttype indexToRemove = gindices[p0];
		sindices[indexToRemove - blockDim.x*blockIdx.x] = 0;
	}

	__syncthreads();

	bool flagSet = (sindices[idx] != 0);

	inttype scan = scanBlock<Plus, exclusive>(sindices, threadIdx.x);

	if (flagSet){
		output[blockIdx.x*blockDim.x - p0 + scan] = input[blockIdx.x*blockDim.x + idx];
	}
}

template <typename T, typename inttype>
__global__
void bulkInsertKernel(T * output, T  const * input,
		              size_t sizeInput,
		              int const * partitions,
		              inttype const * indices,
		              T const * values,
		              size_t sizeIndices)
{
	extern __shared__ int mask[];
	int block = blockIdx.x;
	int idx = threadIdx.x;

	int a0 = partitions[block];
	int a1 = partitions[block+1];

	auto numToInsert = a1 - a0;
	auto numToCopy = BLOCK_SIZE - numToInsert;

	if (idx < numToInsert){

	}



}

template <typename inttype>
__global__
void findPartitionKernel(inttype * output, size_t numOut, inttype const * indices, size_t const numIndices, size_t const sizeX)
{
	//
	inttype blockId = threadIdx.x + blockDim.x * blockIdx.x;
	inttype key = min(blockId*BLOCK_SIZE, sizeX);

	// find id in keys with a binary search
	int a = 0;
	int b = numIndices;

	while (a < b){
		inttype mid = (a+b)/2;
		inttype key2 = indices[mid];

		if (key <= key2){
			b = mid;
		} else {
			a = mid + 1;
		}
	}
	output[blockId] = a;
}

// Look on the diat-th diagonal the point of crossing i where a[i] >= b[i]
template<typename T>
__inline__ __device__
int diagonalSearch(T * a, int aCount, int bCount, int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);

    while(begin < end) {
        int mid = (begin + end)>> 1;
        T aKey = a[mid];
        T bKey = diag - 1 - mid;
        if (aKey <= bKey){
        	begin = mid+1;
        } else {
        	end = mid;
        }
    }
    return begin;
}

template<typename T>
__global__
void findBalancedPartitionKernel(T * dPartitions, int const numParitions, T const * indices, int const sizeIndices, int const sizeX)
{
	int partitionID = threadIdx.x + blockIdx.x * blockDim.x;
	int diagonalPosition = BLOCK_SIZE * partitionID;

	if (partitionID < sizeIndices){
		int cross = diagonalSearch(indices, sizeIndices, sizeX, min(diagonalPosition, sizeX+sizeIndices));
		printf("%d %d %d\n", partitionID, diagonalPosition, cross);

		dPartitions[partitionID] = cross;
	}

}

template <typename inttype, typename T>
gpuVector<inttype> findBalancedPartitions(gpuVector<inttype> const& indices, gpuVector<T> const& X)
{
	auto sizeX = size_t(X.size());
	int numPartitions = divUp(sizeX + indices.size(), BLOCK_SIZE);
	int numBlocks = divUp(numPartitions+1, BLOCK_SIZE);
	std::cout<<numPartitions<<std::endl;
	std::cout<<numBlocks<<std::endl;
	auto partitions = gpuVector<inttype>(numPartitions+1);
	findBalancedPartitionKernel<<<numBlocks, BLOCK_SIZE>>>(partitions.data(), numPartitions+1, indices.data(), indices.size(), sizeX);

	return partitions;
}

template <typename inttype>
gpuVector<inttype> findPartitions(gpuVector<inttype> const& indices, size_t sizeX)
{
	auto numPartitions = divUp(sizeX, BLOCK_SIZE);

	// Create the partitions
	auto partitions = gpuVector<inttype>(numPartitions + 1, indices.size());

	// We need numBlocks
	auto const numBlocks = divUp(numPartitions, BLOCK_SIZE);

	// Binary search to find the partitions
	findPartitionKernel<<<numBlocks, BLOCK_SIZE>>>(partitions.data(), partitions.size(), indices.data(), indices.size(), sizeX);

	return partitions;
}

template <typename T, typename inttype>
gpuVector<T> bulkRemove(gpuVector<T> const& X, gpuVector<inttype> const& I)
{
    auto const numIndicesToRemove = I.size();

	auto const N = X.size();

	auto P = findPartitions(I, N);

	auto numBlocks = divUp(N, BLOCK_SIZE);

    auto Y = gpuVector<T>(N-numIndicesToRemove, size_t(0));

	bulkRemoveKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(Y.data(), X.data(), N, P.data(), I.data(), numIndicesToRemove);

    return Y;

}


template <typename T, typename inttype>
gpuVector<T> bulkInsert(gpuVector<T> const& X, gpuVector<inttype> const& I, gpuVector<inttype> const& V)
{
    auto const numIndicesToInsert = I.size();

	auto const N = X.size();

	auto P = findBalancedPartitions(I, X);

	auto numBlocks = divUp(N + numIndicesToInsert, BLOCK_SIZE);

    auto Y = gpuVector<T>(N + numIndicesToInsert, size_t(0));

	bulkInsertKernel<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(Y.data(), X.data(), N, P.data(), I.data(), V.data(), numIndicesToInsert);

    return Y;
}
#endif
