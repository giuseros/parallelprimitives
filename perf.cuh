#include <cuda.h>

template <typename F, typename... Args>
float gputimeit(const F& func, Args&&... args)
{
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	func(args...);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time;
}
