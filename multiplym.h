#pragma once
#include <vector>
#include <list>

//multiplym.h
namespace MM
{
    class NN
    {
        // Counter for how many models have been trained. Used for setting modelNr
        static unsigned int count;
        // An integer that identifies which model this object is
        const unsigned int modelNr;
        // Input
        std::vector<double> input;
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
        void fprop(const std::vector<double> &in);
        //Backpropagation function. Takes target output as its parameter
        void bprop(const std::vector<double> targetoutput);

        //Softmax function
        void softmax(std::vector<double> &in, std::size_t size);

        //Matrix multiplication function for calculating the product of nodes * weights.
        //The parameter named "left" is the vector of nodes that are multiplied by the weights, or, "right"
        std::vector<double> lmultiply(const std::vector<double> &left, const std::vector<double> &right) const;
        /*Matrix multiplication function for calculating weights during backpropagation.
        The parameter named "left" is the layer of nodes closest to the input, "right" is the error*/
        std::vector<double> wmultiply(const std::vector<double> &left, const std::vector<double> &right) const;
        //Function for updating the parameters
        void updateParameters(const std::vector<double> &dih1w, const std::vector<double> &dh1outw, double dbias1, double dbias2);

        //Function for summing the values of a vector and returning it
        double bsum(const std::vector<double> &v) const;

        //One Hot -encoding function. Gives a corresponding one-hot -vector for a given integer 0-9
        std::vector<double> toOneHot(int v);

        //ReLU-activation function. Returns 0 if d<0, returns the value of d otherwise.
        double relu(double d) const;
        //Function for calculating the derivates of relu for a vector
        std::vector<double> drelu(const std::vector<double> &vec) const;

        public:
        
        NN(std::vector<double> inputs, int h1size, int outsize);
        //Training function
        void train(const std::list<std::tuple<std::vector<double>, std::vector<double>>> &trainingdata);
        //Function that saves the trained parameters and chosen hyperparameters onto disk
        bool saveModel();

    };
}