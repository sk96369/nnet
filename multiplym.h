#include <vector>

//multiplym.h
namespace MM
{
    class NN
    {
        // Input
        std::vector<double> input;
        // Hidden layers
        std::vector<double> h1;
        // Output
        std::vector<double> out;
        //Biases
        double bias1;
        double bias2;

        //Weights
        std::vector<double> wi;
        std::vector<double> w1;

        //Forward propagation function
        void fprop();
        //Backpropagation function. Takes target output as its parameter
        void bprop(const std::vector<double> targetoutput);

        //Matrix multiplication function
        std::vector<double> mmultiply(std::vector<double> left, std::vector<double> right);

        //One Hot -encoding function. Gives a corresponding one-hot -vector for a given integer 0-9
        std::vector<double> toOneHot(int v);

        //ReLU-activation function. Returns 0 if d<0, returns the value of d otherwise.
        double relu(double d);

        public:
        NN(std::vector<double> inputs, int h1size, int outsize);
        //Training function
        void train();

    };
}