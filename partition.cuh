#ifndef PARTITION_CUH__
#define PARTITION_CUH__

#include "iterators.cuh"

// Look on the diag-th diagonal the point of crossing i where a[i] >= b[i]
template<typename it1, typename it2>
__inline__ __device__
int diagonalSearch(it1 a, int aCount, it2 b, int bCount, int diag)
{
    int begin = max(0, diag - bCount);
    int end = min(diag, aCount);

    while(begin < end) {

        int mid = (begin + end)>> 1;

        typename iterator_traits<it1>::value_type aKey = a[mid];
        typename iterator_traits<it2>::value_type bKey = b[diag - 1 - mid];
        if (aKey <= bKey){
        	begin = mid+1;
        } else {
        	end = mid;
        }
    }
    return begin;
}

template<typename T, typename it1, typename it2>
__global__
void findBalancedPartitionKernel(T * dPartitions, int const numParitions, it1 firstRange, int const sizeFirstRange, it2 secondRange, int const sizeSecondRange)
{
	int partitionID = threadIdx.x + blockIdx.x * blockDim.x;
	int diagonalPosition = BLOCK_SIZE * partitionID;
	if (partitionID < sizeFirstRange){
		int diag = min(diagonalPosition, sizeFirstRange + sizeSecondRange);
		int cross = diagonalSearch(firstRange, sizeFirstRange, secondRange, sizeSecondRange, diag);
		dPartitions[partitionID] = cross;
	}

}

#endif
