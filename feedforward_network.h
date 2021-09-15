#pragma once

#include "mm_math.h"
#include <vector>

namespace MM
{
	class nnet
	{

		double learningrate;
		//Input shape: number of features(f) = number of rows, batch size(m) = number of
		//columns.
		mat<int> input;
		//Inputs normalized
		mat<double> input_normalized;
		//Variable determining what the inputs should be normalized based on
		int features_maxvalue;
		//Hidden layer shape: f = number of rows, m = number of columns
		std::vector<mat<double>> hidden_layers;
		//Outputs shape: f = number of rows, m = number of columns
		//This vector stores the activated values of the hidden layers.
		//The last index holds the predictions of the model
		std::vector<mat<double>> outputs;
		//Weights shape: Number of rows = number of next layer's features,
		//               Number of columns = number of previous layer's features
		std::vector<mat<double>> weights;
		//Biases shape: Number of rows = number of the same layer's features
		//              Number of columns = 1
		std::vector<mat<double>> biases;
		//Size signifies the number of layers, excluding the input layer
		int size;

		//Forward propagation
		void fprop();
		//Backpropagation
		void bprop(const std::vector<int> &labels);
		//Function that updates the parameters based on the given delta matrices
		void updateParameters(std::vector<mat<double>> weights_delta, std::vector<mat<double>> biases_delta);

		public:
		//Functions that return a reference to the matrix of the ith layer
		//Returns the input layer if i == -1
		mat<double>& getLayer(int i);
		const mat<double>& getLayer(int i) const;
		//Functions that return a reference to the matrix of the ith output layer
		//Returns the input layer if i == -1
		mat<double>& getOutput(int i);
		const mat<double>& getOutput(int i) const;
		//Function that resets all parameters to random values, as in the constructor
		void resetParameters();

		//Loads the trained parameters from a file named "filename".txt
		void loadModel(std::string filename);
		//Prints the values at layer i to output o, with precision numbers after decimal point
		void printLayer(int i, std::ostream &o, int precision = 1, int imagewidth = 0);

		//Constructor that receives a vector of integers as its argument.
		//The number of elements in the vector determines the number of layers in the
		//network (input and output layers included). The value of each element
		//determines the number of columns in each matrix.
		nnet(const std::vector<int> &dimensions, int f_max, double learningrate);
		//Constructor that constructs this object based on the data written in the file
		//[filename].txt
		nnet(std::string filename);
		//Default constructor
		nnet();

		void train(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int datasize, int epoch);
		//Training function that randomizes the order of the training samples each epoch
		void trainRandom(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int datasize, int epoch, bool printLabels, int imagewidth);
		//Creates a new normalized input based on the given vector
		void setInput(const std::vector<int> &newinput);
		//Function that sets the given vector as input, propagates the data forward and
		//returns predicted labels as a vector
		std::vector<int> predict(const std::vector<int> &data);
		//Saves the trained parameters to a file named "filename".txt
		int saveModel(std::string filename);
		
		/* Getters, setters
		 */
		std::vector<int> getDimensions();
		int getDimension(int i) {return getOutput(i).size();}
		mat<double> getOutput();
	};

}
