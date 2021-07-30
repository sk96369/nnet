#pragma once
#include <string>
#include <vector>
#include <list>

//multiplym.h
namespace MM
{

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
        void bprop(const matrix<int> &targetoutput);

	//Forward propagation function without parameters
	void fprop();

        //Function for updating the parameters
        void updateParameters(const std::vector<double> &dih1w, const std::vector<double> &dh1outw, double dbias1, double dbias2);

        public:
 
	//Function for setting inputs. Returns false if the input vector is of wrong size,
	//true otherwise.
//	bool setInput(const std::vector<int> &in);      
	NN(int inputsize, int h1size, int outsize, std::string filename);
        NN(int h1size, int outsize);
        //Training function
        void train(matrix<int> labelmatrix, int batch_size, int epoch);
        //Function that saves the trained parameters and chosen hyperparameters onto disk
	bool saveModel(std::string filename) const;
	//Function that makes a prediction based on the given inputs
	int predict(const std::vector<int> &in);
    };
}
