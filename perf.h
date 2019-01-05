#include <cuda.h>
#include <ctime>

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

template <typename F, typename... Args>
float cputimeit(const F& func, Args&&... args)
{
	std::clock_t    start;
	start = std::clock();
	func(args...);
    auto t = (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000);
	return double(t);
}
