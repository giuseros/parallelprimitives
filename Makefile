all:
	nvcc -g -std=c++11 main.cu -gencode arch=compute_30,code=\"sm_30,compute_30\" -o main
