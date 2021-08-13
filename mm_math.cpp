#include "mm_math.h"
#include <vector>

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
//		std::cout << "Scalar product original: " << original.toString() << std::endl << "Scalar value: " << scalar << std::endl;
		mat<double> product(original);
		for(auto& i : product.m)
		{
			for(auto& j : i)
			{
				j *= scalar;
			}
		}
//		std::cout << "Scalar product test: " << product.toString() << std::endl;
//		std::cin.get();
		return product;
	}
}

