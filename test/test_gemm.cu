#include "operators.cuh"
#include "gpumatrix.hpp"
#include "gemm.cuh"

#include <vector>
#include <iostream>
#include <numeric>

using namespace pp;
using namespace std;

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> const &hVec)
{
	for (auto e : hVec)
	{
		out << e << " ";
	}
	return out;
}

void simple_test_gemm()
{
    auto A = Matrix<float>(10, 10);
    auto B = Matrix<float>(10, 10);
    for (int i = 0; i<10; i++){
        for (int j = 0; j<10; j++){
            A.set(i, j, i);
            B.set(i, j, i);
        }
    }
    print_matrix(A);
    print_matrix(B);
    auto dA = make_dmatrix(A);
    auto dB = make_dmatrix(B);
    auto dC = gemm(dA,dB);
    auto C = extract_matrix(dC);
    print_matrix(C);
}

int main(){
	cudaDeviceProp prop;

	cudaSetDevice(0);

	cudaGetDeviceProperties(&prop, 0);
	std::cout<<prop.name<<std::endl;

	std::cout<<"Using device "<<0<<":"<<prop.name<<std::endl;

	simple_test_gemm();

    return 0;
}
