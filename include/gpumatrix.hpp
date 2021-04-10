#ifndef GPUMATRIX__H__
#define GPUMATRIX__H__

#include <iostream>
#include <vector>

namespace pp
{

// Host matrix
template <typename T>
class Matrix
{
public:
	Matrix(int cols, int rows): _flat(cols*rows), _cols(cols), _rows(rows) {}
	float at(int x, int y) const{ return _flat[x + _cols*y];}
	void set(int x, int y, float val){  _flat[x + _cols*y] = val;}
	T *data(){return _flat.data();}
	int num_cols() const{return _cols;};
	int num_rows() const{return _rows;};
private:
	std::vector<float>_flat;
	int _cols;
	int _rows;
};

template <typename T>
void print_matrix(Matrix<T> &a)
{
	for (int j = 0; j<a.num_rows(); j++)
	{
		for (int i = 0; i < a.num_cols(); i++)
		{
			std::cout<<a.at(i, j) <<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}

// Device matrix
template <typename T>
struct DMatrix
{
	int num_cols;
	int num_rows;
	T *data;
};

template <typename T>
DMatrix<T> make_dmatrix(Matrix<T>& M)
{
	DMatrix<T> dM;
	dM.num_cols = M.num_cols();
	dM.num_rows = M.num_rows();

	int total_M_size = M.num_cols() * M.num_rows();
	float *dM_ptr;
	cudaMalloc(&dM_ptr, total_M_size*sizeof(T));
	cudaMemcpy(dM_ptr, M.data(), total_M_size*sizeof(T), cudaMemcpyHostToDevice);
	dM.data = dM_ptr;
	return dM;
}

template <typename T>
DMatrix<T> make_dmatrix(int num_cols, int num_rows)
{
	DMatrix<T> dM;
	dM.num_cols = num_cols;
	dM.num_rows = num_rows;
	int total_M_size = num_cols * num_rows;
	float *dM_ptr;
	cudaMalloc(&dM_ptr, total_M_size*sizeof(T));
	dM.data = dM_ptr;
	return dM;
}

template <typename T>
Matrix<T> extract_matrix(const DMatrix<T>& dM)
{
	Matrix<T> M(dM.num_cols, dM.num_rows);
	int total_M_size = dM.num_cols * dM.num_rows;
	cudaMemcpy(M.data(), dM.data, total_M_size*sizeof(T), cudaMemcpyDeviceToHost);
	return M;
}
template <typename T>
__device__ DMatrix<T> sub_matrix_view(const DMatrix<T>& A, int row, int col, int sub_rows, int sub_cols)
{
   DMatrix<T> Asub;
   Asub.num_cols  = sub_cols;
   Asub.num_rows  = sub_rows;
   Asub.data = &A.data[A.num_cols * sub_rows * row + sub_cols * col];
   return Asub;
}


} // namespace pp

#endif