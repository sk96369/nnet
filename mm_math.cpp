#include "mm_math.h"
#include <iostream>
#include <vector>
#include <type_traits>
#include <random>

namespace MM
{
	template <typename A, typename B>
	mat<double> mm(const mat<A> &left, const mat<B> &right)
	{
		if(left.columns() != right.rows())
			return mat<double>(0, 0, 0.0);

		mat<double> newmatrix(right.columns(), left.rows(), (double)0.0);
		for(int i = 0;i<newmatrix.rows();i++)
		{
			for(int j = 0;j<newmatrix.columns();j++)
			{
				for(int k = 0;k < left.columns();k++)
				{
					newmatrix.m[i][j] += left.m[i][k] * right.m[k][j];
				}
			}
		}
		return newmatrix;
	}
		
/*
	template <typename A, typename B>
	mat<double> dot(const mat<A> &left, const mat<B> &right)
	{
		//Start by copying the right side argument
		mat<double> product(right);

		for(auto& i : product.m)
		{
			for(auto& j : i)
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
*/
/*			MAYBE LATER
	template <typename A, typename B>
	mat<double> dot(const mat<A> &left, const mat<B> &right)
	{
		//Start by copying the right side argument
		mat<double> product(right);

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
	*/

	template <typename A, typename B>
	mat<double> hadamard(const mat<A> &left, const mat<B> &right)
	{
		if(left.rows() == right.rows() && left.columns() == right.columns())
		{
			mat<double> product(left);
			for(int i = 0;i<left.rows();i++)
			{
				for(int j = 0;j<left.columns();j++)
				{
					product.m[i][j] *= (double)right.m[i][j];
				}
			}
			return product;
		}
		std::cout << "Incorrect vector sizes, returning an empty vector...\n";
		return mat<double>(0, 0, 0);
	}

	template<typename B>
	mat<B>::mat(double a, double b, int x, int y) : x_s(x), y_s(y)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> distr(a, b);
		m = std::vector<std::vector<double>>(y);
		for(auto& i : m)
		{
			std::vector<double> newline;
			for(int j = 0;j<x;j++)
			{
				newline.push_back(distr(gen));
			}
		}
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
	
	mat(const mat<A> &original) : x_s(original.columns()), y_s(original.rows())
	{
		m = std::vector<std::vector<A>>(original.rows());
		for(int i = 0;i < y_s;i++)
		{
			std::vector<A> newline(original.columns());
			for(int j = 0;j<newline.x_s;j++)
			{
				newline[j] = original.m[i][j];
			}
			m[i] = newline;
		}
	}

	
	template<typename B>
	mat<B> mat<B>::transpose()
	{
		mat<B> transposed(y_s, x_s, m[0][0]);
		for(int i = 0;i < x_s;i++)
		{
			for(int j = 0;j < y_s;j++)
			{
				transposed.m[j][i] = m[i][j];
			}
		}
		return transposed;
	}

	template<typename B>
	std::vector<B> mat<B>::getVector() const
	{
		std::vector<B> vec;
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
		return vec;
	}

	template<typename A>
	mat<double> getSoftmax(mat<A> in)
	{
		return in.softmax();
	}


	void mat::softmax()
	{
		for(auto& in_row : in)
		{
			int size = in.columns();
			double sum = 0;
			for(size_t i = 0;i < size;i++)
			{
				double j = std::exp(in_row[j]);
				sum+=j;
			}
    		
    			for(size_t i = 0;i<size;++i)
    			{
    			    in_row[i] = std::exp(in_row[i])/sum;
    			}
			//Check for a nan-value, and change it to 1 if one is found
			for(auto& i : in_row)
			{
				if(isnan(i))
					i = 1;
			}
		}
	}
	
	bool add(mat<double> &left, mat<double> right)
	{
		std::vector<double> vec = right.getVector();
		int vec_size = vec.size();
		if(vec_size == left.rows());
		{
			for(for int i = 0;i<left.columns())
			{
				for(int j = 0;j<left.rows();j++)
				{
					left.m[j][i] += vec[j];
				}
			}
			return true;
		}
		return false;
	}

	mat<double> getRelu(mat<double> in)
	{
		return in.relu();
	}

	void mat::relu()
	{
		for(auto& i : m)
		{
			for(auto& j : i)
			{
				if(j < 0)
					j = 0;
			}
		}
	}

	mat<int> drelu(mat<double> &matrix)
	{
		mat<int> derivatives(matrix);
		for(auto& i : vec)
		{
			for(auto& j : i)
				j = j > 0;
		}
		return d;
	}

	template<typename A, typename B>
	mat<double> getError(const mat<A> &left, const mat<B> &right)
	{
		mat<double> error((double) 0.0, left.columns(), left.rows());
		for(int i = 0;i<error.columns();i++)
		{
			for(int j = 0;j<error.rows();j++)
			{
				error[i][j] = left[i][j] - right[i][j];
			}
		}
		return error;
	}

	template<typename A>
	mat<double> getTranspose(const mat<A> &original)
	{
		mat<double> transposed(original);
		transposed.transpose();
		return transposed;
	}

	template<typename A>
	mat<A> scalar_m(mat<A> original, double scalar)
	{
		mat<A> product(original);
		for(auto& i : product.m)
		{
			for(auto& j : i)
			{
				j *= scalar;
			}
		}
		return product;
	}
}
