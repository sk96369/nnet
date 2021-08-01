#pragma once
#include <string>
#include <vector>
#include <list>
#include "mm_math.h"

//multiplym.h
namespace MM
{

    class NN
    {
        // Input
        mat<int> input;
        // Hidden layers
        mat<double> h1;
	mat<double> relu_h1;
	mat<double> h2;
        // Output
        mat<double> out;
        //Biases for the first hidden layer and the output layer respectively
        mat<double> bias1;
        mat<double> bias2;
        //Learning rate
        double learningrate;

        //The weight "matrices", saved as 1-D vectors
        mat<double> wi;
        mat<double> w1;

        //Forward propagation function
        void fprop(const mat<int> &in);
        //Backpropagation function. Takes target output as its parameter
        void bprop(const mat<int> &targetoutput);

	//Forward propagation function without parameters
	void fprop();

        //Function for updating the parameters
        void updateParameters(const mat<double> &dih1w, const mat<double> &dh1outw, const mat<double> dbias1, const mat<double> dbias2);

        public:
 
	//Function for setting inputs. Returns false if the input vector is of wrong size,
	//true otherwise.
//	bool setInput(const std::vector<int> &in);      
//	NN(int inputsize, int h1size, int outsize, std::string filename);
	template<typename A>
        NN(mat<A> in, int h1size, int outsize, int batch_size);
        //Training function
        void train(std::vector<int> labelmatrix, int batch_size, int epoch);
        //Function that saves the trained parameters and chosen hyperparameters onto disk
	bool saveModel(std::string filename) const;
	//Function that makes a prediction based on the given inputs
	std::vector<int> predict(const mat<int> &in);
    };
}
