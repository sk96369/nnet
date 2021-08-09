#include "mm_math.h"

namespace MM
{
	class nnet
	{
		double learningrate;
		mat<int> input;
		std::vector<mat<double>> hidden_layers;
		std::vector<mat<double>> weights;
		std::vector<mat<double>> biases;
		mat<double>output;
		int size;
		void fprop();
		void bprop();

		mat<A>& operator[](int i);
		const mat<A>& operator[](int i) const;

		public:
		//Constructor that receives a vector of integers as its argument.
		//The number of elements in the vector determines the number of layers in the
		//network (input and output layers included). The value of each element
		//determines the number of columns in each matrix.
		nnet(std::vector<int> dimensions);
		void train(mat<int> images, std::vector<int> labels, int batchsize);
		
		/* Getters, setters
		 */
		std::vector<int> getDimensions();
	};

}
