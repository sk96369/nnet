#include "mm_math.h"
#include <iostream>
#include <vector>
#include <type_traits>

namespace MM
{
	template <typename A, typename B>
	mat<double> mm(const mat<A> &left, const mat<B> &right)
	{
		if(left.collumns() != right.rows())
			return mat<double>(0, 0, 0.0);

		mat<double> newmatrix(right.collumns(), left.rows(), (double)0.0);
		for(int i = 0;i<left.collumns();i++)
		{
			for(int j = 0;j<left.rows();j++)
			{
				newmatrix.m[i][j] += left.m[i][j] * right.m[j][i];
			}
		}
		return newmatrix;
	}
		

	template <typename A, typename B>
	double dot(const mat<A> &left, const mat<B> &right)
	{
		std::vector<double> left_values = left.getVector();
		std::vector<double> right_values = right.getVector();
		double sum = 0;
		if(left_values.size() == right_values.size())
		{
			int size = left_values.size();
			for(int i = 0;i<size;i++)
			{
				sum+=left_values[i] * right_values[i];
			}
			return sum;
		}
		std::cout << "Incorrect vector sizes, returning 0...\n";
		return sum;

	}

	template <typename A, typename B>
	std::vector<double> hadamard(const mat<A> &left, const mat<B> &right)
	{
		std::vector<double> product;
		std::vector<double> left_values = left.getVector();
		std::vector<double> right_values = right.getVector();
		if(left_values.size() == right_values.size())
		{
			int size = left_values.size();
			for(int i = 0;i<size;i++)
			{
				product.push_back(left_values[i] * right_values[i]);
			}
			return product;
		}
		std::cout << "Incorrect vector sizes, returning an empty vector...\n";
		return product;
	}

	template <typename B>
	mat<B>::mat(B b, int x, int y) : x_s(x), y_s(y)
	{
		m = std::vector<std::vector<B>>(y);
		for(auto& i : m)
		{
			i = std::vector<B>(x, b);
		}
	}

	template <typename B>
	mat<B>::mat(std::vector<B> vec, int x, int y) : x_s(x), y_s(y)
	{
		m = std::vector<std::vector<B>>(y);
		int j = 0;
		int size = vec.size();
		for(int i = 0;i < size;i++)
		{
			if(j%x == 0)
				m.push_back(std::vector<B> newline(x));
			newline[j] = vec[i];
			j++;
		}

	}
	
	template<typename A>
	std::vector<std::vector<A>> mat<A>::transpose()
	{
		mat<A> transposed(y_s, x_s, m[0][0]);
		for(int i = 0;i < x_s;i++)
		{
			for(int j = 0;j < y_s;j++)
			{
				transposed.m[j][i] = m[i][j];
			}
		}
		return transposed;
	}

	template<typename A>
	std::vector<A> mat<A>::getVector() const
	{
		std::vector<A> vec;
		if(x_s > 1)
		{
			if(y_s > 1)
				return vec;
			for(auto& i : vec.m[0])
				vec.push_back(i);
		}
		if(y_s > 1)
		{
			if(x_s > 1)
				return vec;
			for(auto& i : vec.m)
				vec.push_back(i[0]);
		}
	}
	void softmax(std::vector<double> &in, std::size_t size)
	{
    		double sum = 0;
		for(auto& in_member : in)
		{
			double j = std::exp(in_member);
			sum+=j;
		}
		/*
		for(size_t i = 0;i<size;++i)
    		{
			double j = std::exp(in[i]);
			sum+=j;
			if(std::isinf(j) || sum == DBL_MAX)
			{
				for(auto& in_member : in)
				{
					in_member/=2;
				}
				//Start the summing process again
				i = 0;
				sum = 0;
			}
		}*/
    		
    		for(size_t i = 0;i<size;++i)
    		{
    		    in[i] = std::exp(in[i])/sum;
    		}
		//Check for a nan-value, and change it to 1 if one is found
		for(auto& i : in)
		{
			if(isnan(i))
				i = 1;
		}
	}
}
