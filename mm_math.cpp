#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include "headers.h"

namespace MM
{
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

	mat<double> getRelu(mat<double> in)
	{
		in.relu();
		return in;
	}
}
