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
        std::vector<int> input;
        // Hidden layers
        std::vector<double> h1;
        // Output
        std::vector<double> out;
        //Biases for the first hidden layer and the output layer respectively
        std::vector<double> bias1;
        std::vector<double> bias2;
        //Learning rate
        double learningrate;

        //The weight "matrices", saved as 1-D vectors
        std::vector<double> wi;
        std::vector<double> w1;

        //Forward propagation function
        void fprop(const std::vector<int> &in);
        //Backpropagation function. Takes target output as its parameter
        void bprop(const std::vector<int> targetoutput);

	//Forward propagation function without parameters
	void fprop();

	
        //Matrix multiplication functions for calculating the products of nodes * weights.
        //The parameter named "left" is the vector of nodes that are multiplied by the weights, or, "right"
        std::vector<double> lmultiply(const std::vector<double> &left, const std::vector<double> &right) const;
	std::vector<double> lmultiply(const std::vector<int> &left, const std::vector<double> &right) const;


        /*Matrix multiplication function for calculating weights during backpropagation.
        The parameter named "left" is the layer of nodes closest to the input, "right" is the error*/
        std::vector<double> wmultiply(const std::vector<double> &left, const std::vector<double> &right) const;
	std::vector<double> wmultiply(const std::vector<int> &left, const std::vector<double> &right) const;

        //Function for updating the parameters
        void updateParameters(const std::vector<double> &dih1w, const std::vector<double> &dh1outw, double dbias1, double dbias2);

        //Function for summing the values of a vector and returning it
        double bsum(const std::vector<double> &v) const;

        //ReLU-activation function. Returns 0 if d<0, returns the value of d otherwise.
        double relu(double d) const;
        //Function for calculating the derivates of relu for a vector
        std::vector<double> drelu(const std::vector<double> &vec) const;

	//Function for setting inputs. Returns false if the input vector is of wrong size,
	//true otherwise.
	bool setInput(const std::vector<int> &in);

        public:
        
        NN(int h1size, int outsize);
        //Training function, the first member of the tuple is an image, the second is the label
        void train(const std::list<std::tuple<std::vector<int>, int>> &trainingdata, unsigned int iterations = 10);
        //Function that saves the trained parameters and chosen hyperparameters onto disk
	bool saveModel(std::string filename) const;
	//Function that makes a prediction based on the given inputs
	int predict(const std::vector<int> &in);
    };
}
