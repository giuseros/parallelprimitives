/**
 * Copyright (c) 2019 Giuseppe Rossini
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef __UNARY__CUH__
#define __UNARY__CUH__

#include "utils.cuh"
#include "config.cuh"

template <int granularity, typename OP, typename T>
__global__ void unary_kernel(T * in, const int N)
{
	const int idx = threadIdx.x + (blockIdx.x/granularity) * blockDim.x*granularity;

#pragma unroll
	for (int i= 0; i < granularity; i++)
	{
		const int k = idx + i*blockDim.x;
		if (k < N)
		{
			in[k] = OP::apply(in[k]);
		}
	}
}


template <typename OP, typename T>
void unary_op(gpuVector<T>& v)
{
	constexpr int block_size = 1024;
	constexpr int granularity = 11;
	const int num_blocks = divUp(v.size(), block_size*granularity);
	unary_kernel<granularity, OP, T><<<num_blocks, block_size>>>(v.data(), v.size());
}

#endif // __UNARY__CUH__




