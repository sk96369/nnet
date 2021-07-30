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
		int columns() const {return x_s;}
		bool isEmpty() const {return x_s == 0;}
		//A function that returns a vector of all values, if the matrix can be represented
		//as a single vector, otherwise returns an empty vector
		std::vector<A> getVector() const;

		/*Constructors*/
		//Matrix created from an std::vector, split into rows every x members of vec
		mat(std::vector<A> vec, int x = 0, int y = 0);
		//A matrix filled with a
		mat(A a, int x = 0, int y = 0);
		//A matrix filled with random doubles ranging from a to b
		mat(double a, double b, int x, int y);
		//Copy constructor
		template<typename A>
		mat(const mat<A> &original);

		/* Functions defined in .cpp */
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
	mat<double> hadamard(mat<A> left, mat<B> right);

	//Softmax function. Modifies in directly.
	void softmax(mat<double> &in);

	//Addition function. Requires the matrix on the right to be a vector and have the same
	//number of elements as left has columns. Modifies left directly, and returns true on
	//success, false on failure.
	bool add(mat<double> &left, mat<double> right);

	//ReLU function. Inserts a 0 on every element under 0 in the in-matrix
	void relu(mat<double> &in);
	//ReLU derivation function. Returns a matrix including a 0 for each value under 0 in the
	//input matrix, and 1 for each value over 0
	matrix<int> drelu(mat<double> &matrix);

	//Calculates the difference between two matrixes
	template<typename A, typename B>
	matrix<double> getError(const mat<A> &left, const mat<B> &right);

	//Creates a new matrix, that is a transpose of the reference given as argument
	template<typename A>
	matrix<double> getTranspose(const mat<A> &original);

	//Returns a new matrix, with each elements being an element of the original multiplied
	//by the scalar
	template<typename A>
	mat<A> scalar_m(mat<A> &original, double scalar);

	//Creates a vector, so that each element is the sum of each column of the original matrix
	template<typename A>
	mat<A> sum_m(mat<A> &original);
}
