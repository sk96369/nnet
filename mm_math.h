#ifndef MM_MATH_H
#define MM_MATH_H

#include <iostream>
#include <type_traits>
#include <random>
#include <vector>
#include <math.h>

namespace MM
{
	template<typename A>
	class mat
	{
		int x_s;
		int y_s;

		public:
		std::vector<std::vector<A>> m;

		/* Getters, Setters */
		A getType() const {return m[0][0];}
		int rows() const {return y_s;}
		int columns() const {return x_s;}
		bool isEmpty() const {return x_s == 0;}
		//A function that returns a vector of all values, if the matrix can be represented
		//as a single vector, otherwise returns an empty vector
		std::vector<A> getVector() const;

		/*Constructors*/
		//Matrix created from an std::vector, split into rows every x members of vec
		mat();
		mat(const std::vector<A> &vec, int x = 0, int y = 0);
		//A matrix filled with a
		mat(A a, int x = 0, int y = 0);

		//A matrix filled with random doubles ranging from a to b
		mat(double a, double b, int x, int y);
		//Copy constructor
		mat(const mat &o);

		//Function for setting new values for the matrix based on the given vector
		void newValues(const std::vector<A> &vec, int x, int y);
		void newValues(const mat<A> &original);

		/* member functions */
		//Matrix transpose
		void transpose();
		//Applies the softmax function on the second hidden layer to get the output
		//layer.
		void softmax();
		//Applies the relu function
		void relu();

		//toString implementation
		std::string toString();

		//Assignment operator
		const mat & operator=(const mat& matrix);
	};

	template <typename A, typename B>
	mat<double> mm(const mat<A> &left, const mat<B> &right)
	{
		if(left.columns() != right.rows())
		{
			std::cout << "matrix dimension error\nLeft(column row): " << left.columns() << " " << left.rows() << "\nRight(column row): " << right.columns() << " " << right.rows() << std::endl;
			return mat<double>(0, 0, 0.0);
		}
		
		mat<double> newmatrix((double)0.0, right.columns(), left.rows());
		for(int i = 0;i<newmatrix.rows();i++)
		{
			for(int j = 0;j<newmatrix.columns();j++)
			{
				for(int k = 0;k < left.columns();k++)
				{
					newmatrix.m[i][j] += (double)left.m[i][k] * (double)right.m[k][j];
			//		std::cout << newmatrix.m[i][j] << " ";     //TESTOUTPUT
				}
				
			}
		}
/*		for(int i = 0;i<newmatrix.columns();i++)
		{
			for(int j = 0;j<newmatrix.rows();j++)
			{
				std::cout << newmatrix.m[j][i] << " ";
			}
			std::cout << std::endl;
		}
		std::cin.get(); 						TESTOUTPUT*/ 
		return newmatrix;
	}

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
			i=std::vector<double>(x);
		for(int i = 0;i<y;i++)
		{
			for(int j = 0;j<x;j++)
			{
				m[i][j] = distr(rd);
			}
		}
	}

	template <typename B>
	mat<B>::mat(B b, int x, int y) : x_s(x), y_s(y)
	{
		m = std::vector<std::vector<B>>(y);
		for(int i = 0;i < y;i++)
		{
			m[i] = std::vector<B>(x, b);
		}
	}

	template <typename B>
	mat<B>::mat(const std::vector<B> &vec, int x, int y) : x_s(x), y_s(y)
	{
		m = std::vector<std::vector<B>>(y);
		int j = 0;
		int k = 0;
		for(auto& i : m)
		{
			i = std::vector<B>(x);
		}
		for(int i = 0;i < y;i++)
		{
			k++;
		}
		k = 0;
		for(int i = 0;i < y*x;i++)
		{
			m[k][j] = vec[i];
			j++;
			if(j == x)
			{
				j = 0;
				k++;
			}
		}
	}
/*	
	template <typename A>
	mat<A>::mat(const mat &original) : x_s(original.columns()), y_s(original.rows())
	{
		std::cout << "asd" << original.m.size() << " " << original.m[0].size() << std::endl;
		m = std::vector<std::vector<A>>(y_s);
		for(int i = 0;i < y_s;i++)
		{
			std::vector<A> newline(original.columns());
			for(int j = 0;j<x_s;j++)
			{
				newline[j] = original.m[i][j];
			}
			m[i] = newline;
		}
		std::cout << rows() << " " << columns() << std::endl;
	}
*/
	template<typename B>
	void mat<B>::transpose()
	{
		std::vector<std::vector<B>> newm(x_s);
		for(int i = 0;i < x_s;i++)
		{
			newm[i] = std::vector<B>(y_s);
		}
		for(int i = 0;i < x_s;i++)
		{
			for(int j = 0;j < y_s;j++)
			{
				newm[i][j] = m[j][i];
			}
		}
		m = newm;
		x_s = newm[0].size();
		y_s = newm.size();
	}

	template<typename B>
	std::vector<B> mat<B>::getVector() const
	{
		std::vector<B> vec;
		if(x_s > 1)
		{
			if(y_s > 1)
				return vec;
			for(auto& i : m[0])
				vec.push_back(i);
		}
		if(y_s > 1)
		{
			if(x_s > 1)
				return vec;
			for(auto& i : m)
				vec.push_back(i[0]);
		}
		return vec;
	}

	template<typename A>
	mat<double> getSoftmax(mat<A> in)
	{
		in.softmax();
		return in;
	}

	template<typename A>
	void mat<A>::softmax()
	{
		for(auto& in_row : m)
		{
			double sum = 0;
			for(size_t i = 0;i < x_s;i++)
			{
				double j = std::exp(in_row[i]);
				sum+=j;
			}
    		
    			for(size_t i = 0;i<x_s;++i)
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
	
	

	template <typename A>
	mat<double> sum_m(const mat<A> &original)
	{
		mat<double> collapsed(0.0, original.columns(), 1);
		for(int i = 0;i<original.columns();i++)
		{
			double sum = 0.0;
			for(int j = 0;j<original.rows();j++)
			{
				sum+=original.m[j][i];
			}
			collapsed.m[0][i] = sum;
		}
		return collapsed;
	}

	template<typename A>
	void mat<A>::relu()
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

	template <typename B>
	mat<B> drelu(const mat<B> &matrix)
	{
		mat<B> derivatives = matrix;
		for(auto& i : derivatives.m)
		{
			for(auto& j : i)
				j = j > 0;
		}
		return derivatives;
	}

	template<typename A, typename B>
	mat<double> getError(const mat<A> &left, const mat<B> &right)
	{
		if(left.columns() != right.columns() && left.rows() != right.rows())
		{
			std::cout << "Dimension error in getError()\n";
		}
		mat<double> error((double) 0.0, left.columns(), left.rows());
		for(int i = 0;i<error.columns();i++)
		{
			for(int j = 0;j<error.rows();j++)
			{
				error.m[j][i] = (double)left.m[j][i] - right.m[j][i];
			}
		}
		return error;
	}

	template<typename A>
	mat<A> getTranspose(const mat<A> &original)
	{
		mat<A> transposed(original);
		transposed.transpose();
		return transposed;
	}

	template<typename A>
	mat<A> scalar_m(mat<A> original, double scalar)
	{
		mat<A> product = original;
		for(auto& i : product.m)
		{
			for(auto& j : i)
			{
				j *= scalar;
			}
		}
		return product;
	}

	template<typename B>
	const mat<B>& mat<B>::operator=(const mat<B>& matrix)
	{
		return *this;
	}

	template<typename B>
	mat<B>::mat(const mat<B> &o) : y_s(o.rows()), x_s(o.columns())
	{
		m = std::vector<std::vector<B>>(y_s);
		for(int i = 0;i < o.y_s;i++)
		{
			m[i] = std::vector<B>(x_s);
			for(int j = 0;j < x_s;j++)
			{
				m[i][j] = o.m[i][j];
				
			}
		}
	}

	template<typename B>
	std::string mat<B>::toString()
	{
		std::string str = "";
		for(auto& i : m)
		{
			for(auto& j : i)
			{
				str += std::to_string(j);
				str += "/";
			}
			str.pop_back();
			str.append("\n");
		}
		return str;
	}

	template<typename A, typename B>
	bool add(mat<A> &left, mat<B> right)
	{
		std::vector<B> vec = right.getVector();
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

	template<typename B>
	mat<B> getRelu(mat<B> in)
	{
		in.relu();
		return in;
	}
	
	template<typename B>
	void mat<B>::newValues(const std::vector<B> &vec, int x, int y)
	{
		x_s = x;
		y_s = y;
		m.clear();
		m = std::vector<std::vector<B>>(y);
		int j = 0;
		int k = 0;
		for(auto& i : m)
		{
			i = std::vector<B>(x);
		}
		for(int i = 0;i < y*x;i++)
		{
			m[k][j] = vec[i];
			j++;
			if(j == x)
			{
				j = 0;
				k++;
			}
		}
	}
	
	template<typename B>
	void mat<B>::newValues(const mat<B> &original)
	{
		x_s = original.columns();
		y_s = original.rows();
		std::vector<std::vector<B>> new_m(y_s);
		for(int i = 0;i < y_s;i++)
		{
			std::vector<B> new_row(x_s);
			new_m[i] = new_row;
			for(int j = 0;j < x_s;j++)
			{
				new_m[i][j] = original.m[i][j];
			}
		}
		m = new_m;
	}

	
}

#endif
