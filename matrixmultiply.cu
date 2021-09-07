#include <stdio.h>
#include "mm_math.h"
#define N 32

MM::mat<double> gpu_mm(const MM::mat<double> &left, const MM::mat<double> &right);

__global__ void gpu_mm_kernel(int l_size, int l_y, double* d_left, int r_size, int r_y, double* d_right, double* d_output)
{
	//Get the global indexes of the left and right arrays
	int rIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int lIdx = blockIdx.y * blockDim.y + threadIdx.y;
	//Make sure the thread is not trying to access out-of-bounds memory
	if(rIdx < r_size && lIdx < l_size)
	{
		//Calculate which cell of the output array is incremented
		int outputIdx = rIdx/r_y + lIdx % l_y;

		//Multiply the elements and increase the matching element in the output array
		d_output[outputIdx] += d_left[lIdx] * d_right[rIdx];
	}
}

MM::mat<double> gpu_mm(const MM::mat<double> &left, const MM::mat<double> &right)
{
	
	if(left.columns() != right.rows())
	{
		printf("matrix dimension error\nLeft(column row): %i %i \nRight(column row): %i %i\n", left.columns(), left.rows(), right.columns(), right.rows());
	}
	else
	{
		int leftArraySize = left.columns() * left.rows();
		int rightArraySize = right.columns() * right.rows();
		int outputArraySize = left.rows() * right.columns();
		int n = left.columns() * left.rows() * right.columns();

		//Declare the arrays and allocate memory for them on the device
		double* d_left;
		double* d_right;
		double* d_output;
		cudaMalloc(&d_left, leftArraySize * sizeof(double));
		cudaMalloc(&d_right, rightArraySize * sizeof(double));
		cudaMalloc(&d_output, outputArraySize * sizeof(double));

		//Copy the values of the matrix object to a c-style array
		double* leftCArray = left.getCArray();
		double* rightCArray = right.getCArray();
		cudaMemcpy(d_left, leftCArray, left.columns() * left.rows() * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_right, rightCArray, right.columns() * right.rows() * sizeof(double), cudaMemcpyHostToDevice);

		//Free the memory from the c-style arrays
		delete[] leftCArray;
		delete[] rightCArray;
		//Set the kernel launch parameters
		dim3 GRID(ceil(right.columns()/32*16), ceil(left.rows()/32*16));
		dim3 BLOCK(32*16, 32*16);
		
		//Launch the kernel
		gpu_mm_kernel<<<GRID, BLOCK>>>(leftArraySize, left.rows(), d_left, rightArraySize, right.rows(), d_right, d_output);
		//Move the output array from the device to the host
		double* outputCArray = (double*)malloc(right.columns() * left.rows() * sizeof(double*));
		cudaMemcpy(outputCArray, d_output, right.columns() * left.rows() * sizeof(double), cudaMemcpyDeviceToHost);
		// Construct a matrix object based on the output c-style array
		MM::mat<double> outputmatrix(outputCArray, outputArraySize, left.rows(), right.columns());
		//Free the memory allocated on the gpu
		cudaFree(d_left);
		cudaFree(d_right);
		cudaFree(d_output);
		delete[] outputCArray;
		return outputmatrix;
	}
	//The declaration promises to return something, so we return the left matrix
	return left;
}

