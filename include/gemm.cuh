#ifndef GEMM_HH__
#define GEMM_HH__

#include <cuda.h>
#include <vector>
#include <iostream>

#include "gpumatrix.hpp"
#define GEMM_BLOCK_SIZE 32
#define divUp(x, y) ((x + y - 1) / (y))


namespace pp{
    namespace kernel{

        template <typename T>
        __global__
        void gemm_shared(const DMatrix<T> A, const DMatrix<T> B, DMatrix<T> C)
        {
            // Block row and column
            int blockRow = blockIdx.y;
            int blockCol = blockIdx.x;

            // Each thread block computes one sub-matrix Csub of C
            DMatrix<T> Csub = sub_matrix_view<T>(C,
                                              blockRow,
                                              blockCol,
                                              GEMM_BLOCK_SIZE,
                                              GEMM_BLOCK_SIZE);

            // Each thread computes one element of Csub
            // by accumulating results into D
            T D = 0;

            // Thread row and column within Msub
            int row = threadIdx.y;
            int col = threadIdx.x;
            
            const int global_row = blockRow * GEMM_BLOCK_SIZE + row;
            const int global_col = blockCol * GEMM_BLOCK_SIZE + col;
            const bool outside_scope_C = global_row >= C.num_rows || global_col >= C.num_cols;
            const int num_blocks = (A.num_cols  + GEMM_BLOCK_SIZE - 1) / GEMM_BLOCK_SIZE;

        #pragma unroll
            for (int m = 0; m < num_blocks; ++m) {

                // Get sub-matrix Asub of A
                DMatrix<T> Asub = sub_matrix_view(A, blockRow, m, GEMM_BLOCK_SIZE, GEMM_BLOCK_SIZE );
                const bool outside_scope_A = global_row >= A.num_rows || m*GEMM_BLOCK_SIZE + col >= A.num_cols;

                // Get sub-matrix Bsub of B
                DMatrix<T> Bsub = sub_matrix_view(B, blockCol, m, GEMM_BLOCK_SIZE, GEMM_BLOCK_SIZE);
                const bool outside_scope_B = global_row >= B.num_rows || m*GEMM_BLOCK_SIZE + col >= B.num_cols;

                // Shared memory used to store Asub and Bsub respectively
                __shared__ T As[GEMM_BLOCK_SIZE][GEMM_BLOCK_SIZE];
                __shared__ T Bs[GEMM_BLOCK_SIZE][GEMM_BLOCK_SIZE];

                // Load into shared mem
                As[row][col] = (outside_scope_A ? 0 : Asub.data[row*A.num_cols + col]);
                Bs[row][col] = (outside_scope_B ? 0 : Bsub.data[row*B.num_cols + col]);

                // Synchronize to make sure the sub-matrices are loaded
                // before starting the computation
                __syncthreads();

                // Multiply Asub and Bsub together
                for (int e = 0; e < GEMM_BLOCK_SIZE; ++e)
                {
                    D += As[row][e]*Bs[e][col];
                }

                // Synchronize to make sure that the preceding
                // computation is done before loading two new
                // sub-matrices of A and B in the next iteration
                __syncthreads();
            }

            if (!outside_scope_C)
            {
                Csub.data[row * C.num_rows + col] = D;
            }

        }
    }

    template <typename T>
	DMatrix<T> gemm(const DMatrix<T> &A, const DMatrix<T> &B)
	{
		auto C = make_dmatrix<T>(A.num_rows, B.num_cols);
        dim3 dimBlock(GEMM_BLOCK_SIZE, GEMM_BLOCK_SIZE);
        dim3 dimGrid( divUp(C.num_cols, GEMM_BLOCK_SIZE), divUp(C.num_rows, GEMM_BLOCK_SIZE) );
        kernel::gemm_shared<T><<<dimGrid, dimBlock>>>(A, B, C);
		return C;
	}
} // namespace pp

#endif // GEMM_HH__