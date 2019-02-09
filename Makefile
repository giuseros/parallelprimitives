NAME=test_scan

all:
	nvcc -O3 -std=c++11 -I. src/$(NAME).cu -gencode arch=compute_75,code=\"sm_75,compute_75\" -o  $(NAME)
	#nvcc -g -G -std=c++11 -I. src/test_scan.cu -gencode arch=compute_75,code=\"sm_75,compute_75\" -o test_scan
	
