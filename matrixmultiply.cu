#include <stdio.h>
#include "mm_math.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

MM::mat<double> gpu_mm(const MM::mat<double>& left, const MM::mat<double>& right);

__global__ void gpu_mm_kernel(int M, int N, int w, const double* d_left, const double* d_right, double* d_output)
{
	//Get the global indexes of the left and right arrays
	int lIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int rIdx = blockIdx.y * blockDim.y + threadIdx.y;

	//Make sure the thread is not trying to access out-of-bounds memory
	if (rIdx < N && lIdx < M)
	{
		double value = 0;
		for (int i = 0; i < w; i++)
		{
			value += d_left[lIdx * w + i] * d_right[rIdx + N * i];
		}
		d_output[lIdx * N + rIdx] = value;
	}

}

MM::mat<double> gpu_mm(const MM::mat<double>& left, const MM::mat<double>& right)
{

	if (left.columns() != right.rows())
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
		gpuErrchk(cudaMalloc(&d_left, leftArraySize * sizeof(double)));
		gpuErrchk(cudaMalloc(&d_right, rightArraySize * sizeof(double)));
		gpuErrchk(cudaMalloc(&d_output, outputArraySize * sizeof(double)));

		//Copy the values of the matrix object to a c-style array
		double* leftCArray = left.getCArray();
		double* rightCArray = right.getCArray();
		//Create an output matrix full of zeroes
		double* outputCArray = (double*)calloc(outputArraySize, sizeof(double));

		/*for(int i = 0;i<leftArraySize;i++)
		{
			printf("%f ", leftCArray[i]);
		}
		printf("\n");
		for(int i = 0;i<rightArraySize;i++)
		{
			printf("%f ", rightCArray[i]);
		}
		printf("\n");*/

		gpuErrchk(cudaMemcpy(d_left, leftCArray, leftArraySize * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_right, rightCArray, rightArraySize * sizeof(double), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMemcpy(d_output, outputCArray, outputArraySize * sizeof(double), cudaMemcpyHostToDevice));


		//Free the memory from the c-style arrays
		delete[] leftCArray;
		delete[] rightCArray;


		//Set the kernel launch parameters
		dim3 GRID((int)ceil((double)left.rows() / 32), (int)ceil((double)right.columns() / 32));
		dim3 BLOCK(32, 32);

		//Launch the kernel
		gpu_mm_kernel << <GRID, BLOCK >> > (left.rows(), right.columns(), left.columns(), d_left, d_right, d_output);
		gpuErrchk(cudaPeekAtLastError());

		//Wait for the device to complete its work
		gpuErrchk(cudaDeviceSynchronize());
		//Move the output array from the device to the host
		gpuErrchk(cudaMemcpy(outputCArray, d_output, right.columns() * left.rows() * sizeof(double), cudaMemcpyDeviceToHost));

		/*for(int i = 0;i<outputArraySize;i++)
		{
			printf("%f ", outputCArray[i]);
		}
		printf("\n");*/
		// Construct a matrix object based on the output c-style array
		MM::mat<double> outputmatrix(outputCArray, outputArraySize, left.rows(), right.columns());
		//printf("%s\n", outputmatrix.toStringFlipped(0).c_str());
		//Free the memory allocated on the gpu
		cudaFree(d_left);
		cudaFree(d_right);
		cudaFree(d_output);
		delete[] outputCArray;
		return outputmatrix;
	}
	//The declaration promises to return something, so we return the left matrix in the case of error
	return left;
}
