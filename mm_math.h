#ifndef MM_MATH_H
#define MM_MATH_H

#include <iostream>
#include <type_traits>
#include <random>
#include <vector>
#include <math.h>
#include <cfloat>
#include <iomanip>
#include <sstream>
#include <chrono> //For measuring time taken on functions

namespace MM
{

	/* Usage of this matrix class from feedforward_network:
	 * 	Choose the number of units in the layer and call the constructor with the number
	 * 	(the one below the comment "---node_constructor---")
	 * 	Call the newValues function directly below the comment "---input_values---"
	 * 	giving it a vector of size batch_size*units_in_the_layer as the argument.
	 * 	All done!
	 */

	template<typename A>
	class mat
	{
		int x_s;
		int y_s;
		public:
		std::vector<std::vector<A>> m;

		/* Getters, Setters */
		int size() const {return y_s;}
		int rows() const {return y_s;}
		int columns() const {return x_s;}
		bool isEmpty() const {return x_s == 0;}
		//Calculates random values for the elements of the matrix, as described by Kaiming He (2015)
		//He weight initialization: weight = G(0, sqrt(2/n)), where G is a random number with a Gaussian probability distribution
		//with a mean of 0 and a standard deviation of square root 2/n, where n is the number of input nodes
		void heInitialize(double mean, double stDev);
		//A function that returns a vector of all values, if the matrix can be represented
		//as a single vector, otherwise returns an empty vector
		std::vector<A> getVector() const;
		//A function that returns the equivalent c-style (1-D) array of the matrix
		A* getCArray() const;
		//Transposes the matrix
		void transpose();

		/*Constructors*/
		//Default constructor
		mat();
		//---node_constructor---
		//Constructor that creates given number of empty vectors
		mat(int layersize);
		//Matrix created from an std::vector, split into columns every x members of vec
		mat(const std::vector<A> &vec, int y = 0, int x = 0);
		//Copy the vector of vectors given as arguments into m
		mat(const std::vector<std::vector<A>> &mcopy);
		//A matrix filled with a
		mat(A a, int x, int y);
		//A matrix filled with random doubles ranging from a to b
		mat(double a, double b, int x, int y);
		//Copy constructor
		mat(const mat &o);
		//Constructs a matrix based on the given c-style array
		mat(const A* cArray, int size, int y, int x);

		//Functions for setting new values for the matrix based on the given arguments
		//---input_values---
		void newValues(const std::vector<A> &vec);
		void newValues(const std::vector<A> &vec, int x, int y);
		void newValues(const mat<A> &original);

		/* member functions */
		//Applies the ReLU function to the matrix
		void relu();
		//Applies the leaky ReLU function to the matrix 
		//Takes the multiplier for negative values as its argument. The default multiplier is 0.01
		void leakyRelu(double multiplier = 0.01);

		//toString implementation
		//prints each column horizontally, printed with the given precision value
		//If precision == -1, prints out symbols instead of numbers
		std::string toString(int precision = -1, int imagewidth = 0) const;
		//prints each row horizontally, with possible floating point
		//values printed with the given precision value
		std::string toStringFlipped(int precision = 0) const;

		//Assignment operator
		mat<A> & operator=(const mat& matrix);
		
		std::vector<A>& operator[](int i) {return m[i];}
		const std::vector<A>& operator[](int i) const {return m[i];}
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
//		std::cout << "left matrix: " << left.m.size() << " " << left.m[0].size() << std::endl;
//		std::cout << "right matrix: " << right.m.size() << " " << right.m[0].size() << std::endl;
//		std::cout << "newmatrix: " << newmatrix.m.size() << " " << newmatrix.m[0].size() << std::endl;
		for(int i = 0;i<newmatrix.m.size();i++)
		{
			for(int j = 0;j<newmatrix.m[0].size();j++)
			{
				for(int k = 0;k < left.m.size();k++)
				{
					newmatrix.m[i][j] += (double)left.m[k][j] * (double)right.m[i][k];
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
		std::cout << "Hadamard - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();
		if(left.rows() == right.rows() && left.columns() == right.columns())
		{
			mat<double> product(0.0, left.columns(), left.rows());
			for(int i = 0;i<left.columns();i++)
			{
				for(int j = 0;j<left.rows();j++)
				{
					product.m[i][j] = (double)left.m[i][j] * (double)right.m[i][j];
				}
			}
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << duration.count() << " microseconds\n";
			return product;
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::cout << "Incorrect vector sizes, returning an empty vector...\n";
		return mat<double>(0, 0, 0);
	}

	template<typename B>
	mat<B>::mat(double a, double b, int x, int y) : x_s(x), y_s(y), m(x)
	{
		for (int i = 0; i < x; i++)
		{
			m[i] = std::vector<double>(y);
		}
		//Calculate the standard deviation for He initialization, and call the function
		double stDev = sqrt((double)2 / (double)x_s);
		heInitialize(0.0, stDev);

/*		std::random_device rd;       <---DEPRECATED WEIGHT INITIALIZATION
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> distr(a, b);
		for(int i = 0;i<x;i++)
		{
			m[i] = std::vector<double>(y);
			for(int j = 0;j<y;j++)
			{
				m[i][j] = distr(rd);
			}
		}*/
	}

	template <typename B>
	mat<B>::mat(B b, int x, int y) : x_s(x), y_s(y), m(x)
	{
		for(int i = 0;i < x;i++)
		{
			m[i] = std::vector<B>(y, b);
		}
	}

	template <typename B>
	mat<B>::mat(const std::vector<B> &vec, int x, int y) : y_s(y), x_s(x), m(x)
	{
		int j = 0;
		int k = 0;
		for(auto& i : m)
		{
			i = std::vector<B>(y);
		}
		for(int i = 0;i < y*x;i++)
		{
			m[k][j] = vec[i];
			j++;
			if(j == y)
			{
				j = 0;
				k++;
			}
		}
	}

	template<typename B>
	void mat<B>::transpose()
	{
		std::vector<std::vector<B>> newm(y_s);
		for(int i = 0;i < y_s;i++)
		{
			newm[i] = std::vector<B>(x_s);
		}
		for(int i = 0;i < y_s;i++)
		{
			for(int j = 0;j < x_s;j++)
			{
				newm[i][j] = m[j][i];
			}
		}
		m = newm;
		y_s = newm[0].size();
		x_s = newm.size();
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
	mat<A> softmax(const mat<A> &o)
	{
		std::cout << "softmax - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		mat<A> softmaxed(o);
		for(int i = 0;i<softmaxed.columns();i++)
		{
			double sum = 0;
			for(int j = 0;j < softmaxed.rows();j++)
			{
				sum += std::exp(softmaxed[i][j]);
			}
			for(int j = 0;j<softmaxed.rows();j++)
			{
				softmaxed[i][j] = std::exp(softmaxed[i][j]) / sum;
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";
		return softmaxed;
	}


	template<typename A>
	mat<A> getSoftmax(const mat<A> &o)
	{
		std::cout << "getSoftmax - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		mat<A> softmaxed(o);
		for(int i = 0;i<softmaxed.columns();i++)
		{
			//Calculate the sum of all the values in column i
			double sum = 0;
			for(int j = 0;j < softmaxed.rows();j++)
			{
				sum += std::exp(softmaxed[i][j]);
			}
    		
			//Calculate the softmax for the column based on the sum
    		for(int j = 0;j<softmaxed.rows();j++)
    		{
    		    softmaxed[i][j] = std::exp(softmaxed[i][j])/sum;
    		}
			//Check for a nan-value, and change it to 1 if one is found
			for(int j = 0;j<softmaxed.rows();j++)
			{
				if(isnan(softmaxed[i][j]))
				{
//					std::cout << "nan detected " << sum << std::endl << std::cin.get();
					softmaxed[i][j] = 1;
				}
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";

		return softmaxed;
	}
	
	

	template <typename A>
	mat<double> sum_m(const mat<A> &original)
	{
		std::cout << "sum_m - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		mat<double> collapsed(0.0, 1, original.rows());
		for(int i = 0;i<original.rows();i++)
		{
			double sum = 0.0;
			for(int j = 0;j<original.columns();j++)
			{
				sum+=original.m[j][i];
			}
			collapsed.m[0][i] = sum;
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";

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
	mat<int> drelu(const mat<B> &matrix)
	{
		mat<int> derivatives(0, matrix.columns(), matrix.rows());
		for(int i = 0;i<derivatives.columns();i++)
		{
			for(int j = 0;j<derivatives.rows();j++)
			{
				derivatives[i][j] = matrix[i][j] > 0;
			}
		}
		return derivatives;
	}

	template<typename A, typename B>
	mat<double> getError(const mat<A> &left, const mat<B> &right)
	{
		std::cout << "getError - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		if(left.columns() != right.columns() && left.rows() != right.rows())
		{
			std::cout << "Dimension error in getError()\n";
		}
		mat<double> error((double)0.0, left.columns(), left.rows());
		for(int i = 0;i<error.columns();i++)
		{
			for(int j = 0;j<error.rows();j++)
			{
//				std::cout << "Left matrix[" << i << "][" << j << "] = " << left.m[i][j] << " - " << "Right matrix[" << i << "][" << j << "] = " << right.m[i][j] << " = " << left.m[i][j] - right.m[i][j] << std::endl;
				error.m[i][j] = left.m[i][j] - right.m[i][j];
//				std::cout << "Error[" << i << "][" << j << "] = " << error[i][j] << std::endl;
//				std::cin.get();
			}
		}
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";

		return error;
	}

	template<typename A>
	mat<A> getTranspose(const mat<A> &original)
	{
		mat<A> transposed(original);
		transposed.transpose();
		return transposed;
	}

	

	template<typename B>
	mat<B>& mat<B>::operator=(const mat<B>& matrix)
	{
		x_s = matrix.columns();
		y_s = matrix.rows();
		m.clear();
		m.assign(x_s, std::vector<B>(y_s));
		for(int i = 0;i < x_s;i++)
		{
			for(int j = 0;j<y_s;j++)
			{
				m[i][j] = matrix[i][j];
			}
		}
		return *this;
	}

	template<typename B>
	mat<B>::mat(int layersize) : y_s(layersize), x_s(0)
	{
	}

	template<typename B>
	mat<B>::mat(const mat<B> &o) : y_s(o.rows()), x_s(o.columns()), m(std::vector<std::vector<B>>(x_s))
	{
		for(int i = 0;i < x_s;i++)
		{
			m[i] = std::vector<B>(y_s);
			for(int j = 0;j < y_s;j++)
			{
				m[i][j] = o.m[i][j];
				
			}
		}
	}

	template<typename B>
	std::string mat<B>::toString(int precision, int imagewidth) const
	{
		std::stringstream ss;
		if(precision >= 0)
		{
			ss << std::fixed << std::setprecision(precision);
			for(auto& i : m)
			{
				for(auto& j : i)
				{
					ss << j << " ";
				}
			}
		}
		else
		{
			for(auto& i: m)
			{
				int k = 0;
				for(auto& j : i)
				{
					if(j == 0)
						ss << " ";
					else
						ss << "#";
					k++;
					if(k == imagewidth)
					{
						ss << "\n";
						k = 0;
					}
				}
			}
		}
		return ss.str();
	}

	template<typename B>
	std::string mat<B>::toStringFlipped(int precision) const
	{
		std::stringstream ss;
		ss << std::fixed << std::setprecision(precision);
		for(int i = 0;i<y_s;i++)
		{
			for(int j = 0;j<x_s;j++)
			{
				ss << m[j][i] << " ";
			}
			ss << "\n";
		}
		return ss.str();
	}

	//Function that copies the left matrix, then adds each the value from right
	//to each of the elements on the left, row by row
	template<typename A, typename B>
	mat<double> add(const mat<A> &left, const mat<B> right)
	{
		std::cout << "add - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		mat<double> sum_matrix(left);
		if(right.rows() == left.rows() && right.columns() == 1);
		{
			for(int i = 0;i<left.columns();i++)
			{
				for(int j = 0;j<left.rows();j++)
				{
					sum_matrix[i][j] += right[0][j];
				}
			}
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << duration.count() << " microseconds\n";

			return sum_matrix;
			
		}
		//If nothing has been returned yet, return the copied left matrix and print
		//an error message
		std::cout << "Addition error!\n";
		return sum_matrix;

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
	void mat<B>::newValues(const std::vector<B> &vec)
	{
		x_s = vec.size() / y_s;
		for(int i = 0;i<x_s;i++)
		{
			m[i] = std::vector<B>(y_s);
		}
		for(int i = 0;i<x_s;i++)
		{
			for(int j = 0;j<y_s;j++)
			{
				m[i][j] = vec[i*y_s + j];
			}
		}
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
		m.clear();
		m = original.m;
		for(int i = 0;i < x_s;i++)
		{
			m[i] = original.m[i];
			for(int j = 0;j < y_s;j++)
			{
				m[i][j] = original.m[i][j];
			}
		}
	}

	template<typename B>
	mat<B>::mat(const std::vector<std::vector<B>> &mcopy) : x_s(mcopy.size()), y_s(mcopy[0].size()), m(mcopy.size())
	{
		for(int i = 0;i<x_s;i++)
		{
			std::vector<B> newcolumn(y_s);
			m[i] = newcolumn;
			for(int j = 0;j<y_s;j++)
			{
				m[i][j] = mcopy[i][j];
			}
		}
	}

	template<typename B>
	mat<B>::mat(const B* cArray, int size, int y, int x) : x_s(x), y_s(y), m(x)
	{
		for (int i = 0; i < x_s; i++)
		{
			std::vector<B> newcolumn(y);
			m[i] = newcolumn;
		}
		for (int i = 0; i < size; i++)
		{
			m[i % x][i / x] = cArray[i];
		}
	}

	template <typename B>
	B* mat<B>::getCArray() const
	{
		B* cArray = (B*)malloc(x_s * y_s * sizeof(B));
		for (int i = 0; i < x_s * y_s; i++)
		{
			cArray[i] = m[i % x_s][i / x_s];
		}
		return cArray;
	}

	void leakyRelu(double multiplier)
	{
		for (auto& i : m)
		{
			for (auto& j : i)
			{
				if (j < 0)
					j = multiplier * j;
			}
		}
	}

	mat<double> getNormalized(const mat<int> &original, int feature_max);
	mat<double> scalar_m(const mat<double> &original, double scalar);
}

#endif
