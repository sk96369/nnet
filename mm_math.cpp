#include "mm_math.h"
#include <vector>
#include <chrono> //For measuring time taken on functions

namespace MM
{
	/*
	bool add(mat<double> &left, mat<double> right)
	{
		std::vector<double> vec = right.getVector();
		int vec_size = vec.size();
		if(vec_size == left.rows());
		{
			for(int i = 0;i<left.columns();i++)
			{
				for(int j = 0;j<left.rows();j++)
				{
					left.m[j][i] += vec[j];
				}
			}
			return true;
		}
		std::cout << "Addition error!\n";
		return false;
	}

	template<typename B>
	mat<B>::mat() : x_s(0), y_s(0), m(std::vector<std::vector<B>>(0))
	{
	}

	mat<double> getRelu(mat<double> in)
	{
		in.relu();
		return in;
	}
	*/
	mat<double> getNormalized(const mat<int> &original, int feature_max)
	{
		std::vector<std::vector<double>> new_m(original.columns());
		for(int i = 0;i < original.columns();i++)
		{
			std::vector<double> new_column(original.rows());
			new_m[i] = new_column;
			for(int j = 0;j < original.rows();j++)
			{
				new_m[i][j] = (double)original[i][j] / (double)feature_max;
			}
		}
		mat<double> normalized(new_m);
		return normalized;
	}

	mat<double> scalar_m(const mat<double> &original, double scalar)
	{
		std::cout << "scalar_m - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		//std::cout << "Scalar product original:\n" << original.toString(2) << std::endl;
//		std::cout << "Scalar: " << scalar << std::endl;
//		std::cout << "Scalar original matrix: \n" << original.toString(1) << std::endl;
		mat<double> product(original);
//		std::cout << "Copy of the original matrix: \n" << product.toString(1) << std::endl;
		for(auto& i : product.m)
		{
			for(auto& j : i)
			{
				j *= scalar;
			}
		}
//		std::cout << "Scalar product: \n" << product.toString(1) << std::endl;
//		std::cout << "Scalar product inside the scalar_m function:\n" << product.toString(2) << std::endl;
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";
		
		return product;
	}
}

