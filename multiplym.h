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

        //Weights
        std::vector<double> wi;
        std::vector<double> w1;

        //Forward propagation function
        void fprop();

        //Matrix multiplication function
        std::vector<double> mmultiply(std::vector<double> left, std::vector<double> right);

        //ReLU-activation function. Returns 0 if d<0, returns the value of d otherwise.
        double relu(double d);

        public:
        NN::NN(int isize, int h1size, int outsize);

    };
}