#ifndef MERGE_HH__
#define MERGE_HH__

#include "partition.cuh"

template <typename T>
__global__ void
mergeKernel(T * merged, int const * partitions, T const * a, int const sizeA, T const * b, int const sizeB)
{
	extern __shared__ T buffer[];

	int block = blockIdx.x;
	int tid = threadIdx.x;

	int a0 = partitions[block];
	int a1 = partitions[block+1];
	int b0 = BLOCK_SIZE*block - a0;

	int nA = a1 - a0;
	int nB = min(int(BLOCK_SIZE - nA), sizeB-b0);

	if (tid < nA && tid < nB){
		buffer[tid] = a[block * BLOCK_SIZE + tid -a0];
		buffer[tid+nA] = b[block * BLOCK_SIZE + tid -b0];
	} else if (tid < nA){
		buffer[tid] = a[block *BLOCK_SIZE + tid];
	} else {
		buffer[tid+nA] = b[block *BLOCK_SIZE + tid];
	}

	__syncthreads();


	// Find your position with a diagonal search
	int idxToInsertA = diagonalSearch(buffer, nA, buffer + nA, nB, tid);
	int idxToInsertB = (tid - idxToInsertA) + nA;
	if (block==0 && tid ==0){
		printf("%d\n", a0);
		printf("%d\n", a1);
//		printf("%d\n", b0);
//		printf("%d\n", b1);
		for (int i = 0;i<BLOCK_SIZE;i++){
			printf("%d ", int(buffer[i]));
		}
		printf("\n");
		printf("%d %d\n", idxToInsertA, int(buffer[idxToInsertA]));
		printf("%d %d\n", idxToInsertB, int(buffer[idxToInsertB]));

	}

	if (idxToInsertA < nA && buffer[idxToInsertA] <= buffer[idxToInsertB]){
		merged[block*BLOCK_SIZE + tid] = buffer[idxToInsertA];
	} else {
		merged[block*BLOCK_SIZE + tid] = buffer[idxToInsertB];
	}

}

template<typename T>
gpuVector<T> merge(gpuVector<T> const& a, gpuVector<T> const& b)
{
	auto N = a.size() + b.size();
	int  numPartitions = divUp(N, BLOCK_SIZE);

	gpuVector<int> partitions(numPartitions + 1, 0);

	int const numBlocksPartitions = divUp(numPartitions+1, BLOCK_SIZE);

	findBalancedPartitionKernel<<<numBlocksPartitions, BLOCK_SIZE>>>(partitions.data(), numPartitions+1, a.data(), a.size(), b.data(), b.size());

	gpuVector<T> merged(N);

	int const numBlocksMerge = divUp(N,BLOCK_SIZE);

	mergeKernel<<<numBlocksMerge, BLOCK_SIZE, BLOCK_SIZE*sizeof(T)>>>(merged.data(), partitions.data(), a.data(), a.size(), b.data(), b.size());

	return merged;
}


#endif

