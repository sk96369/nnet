#pragma once
#include <string>
#include <vector>
#include <list>

//multiplym.h
namespace MM
{
	//Softmax function
	void softmax(std::vector<double> &in, std::size_t size);

    class NN
    {
        // Input
        matrix<int> input;
        // Hidden layers
        matrix<double> h1;
        // Output
        matrix<double> out;
        //Biases for the first hidden layer and the output layer respectively
        matrix<double> bias1;
        matrix<double> bias2;
        //Learning rate
        double learningrate;

        //The weight "matrices", saved as 1-D vectors
        matrix<double> wi;
        matrix<double> w1;

        //Forward propagation function
        void fprop(const matrix<int> &in);
        //Backpropagation function. Takes target output as its parameter
        void bprop(const matrix<int> targetoutput);

	//Forward propagation function without parameters
	void fprop();

        //Function for updating the parameters
        void updateParameters(const std::vector<double> &dih1w, const std::vector<double> &dh1outw, double dbias1, double dbias2);

        //ReLU-activation function. Returns 0 if d<0, returns the value of d otherwise.
        double relu(double d) const;
        //Function for calculating the derivates of relu for a vector
        std::vector<double> drelu(const std::vector<double> &vec) const;


        public:
 
	//Function for setting inputs. Returns false if the input vector is of wrong size,
	//true otherwise.
	bool setInput(const std::vector<int> &in);      
	NN(int inputsize, int h1size, int outsize, std::string filename);
        NN(int h1size, int outsize);
        //Training function
        void train(matrix<int> labelmatrix, matrix<int> imagematrix, int batch_size, int iterations);
        //Function that saves the trained parameters and chosen hyperparameters onto disk
	bool saveModel(std::string filename) const;
	//Function that makes a prediction based on the given inputs
	int predict(const std::vector<int> &in);
    };
}
