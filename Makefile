all:
	nvcc -g -std=c++11 test_reduction.cu -gencode arch=compute_60,code=\"sm_60,compute_60\" -o test_reduction
