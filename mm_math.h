#pragma once

#include <vector>

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
		int collumns() const {return x_s;}
		bool isEmpty() const {return x_s == 0;}
		//A function that returns a vector of all values, if the matrix can be represented
		//as a single vector, otherwise returns an empty vector
		std::vector<A> getVector() const;
		/* Functions defined in .cpp */

		//Constructors
		//Matrix created from an std::vector, split into rows every x members of vec
		mat(std::vector<A> vec, int x = 0, int y = 0);
		//A matrix filled with a
		mat(A a, int x = 0, int y = 0);
		//A matrix filled with random doubles ranging from a to b
		mat(double a, double b, int x, int y)
		//Matrix transpose
		std::vector<std::vector<A>> transpose();
	};

	/* Matrix operations */
	//Matrix multiplication
	template <typename A, typename B>
	mat<double> mm(mat<A> left, mat<B> right);

	//Dot product
	template <typename A, typename B>
	double dot(mat<A> left, mat<B> right);

	//Hadamard product
	template <typename A, typename B>
	std::vector<double> hadamard(mat<A> left, mat<B> right);

	//Softmax function
	void softmax(std::vector<double> &in, std::size_t size)
}
