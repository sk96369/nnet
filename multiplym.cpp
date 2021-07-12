#include "multiplym.h"
#include <vector>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <iostream>

//multiplym.cpp
int main()
{
    std::ifstream in;
    in.open("in.txt");
    if(!in)
    {
        std::cout << "Unable to open \"in.txt\"";
        exit(1);
    }

    std::vector<double> inputs;
    double input;
    while(in >> input)
    {
        inputs.push_back(input);
    }
    in.close();

    MM::NN network(inputs.size(), 10, 10);

    return 0;
}

namespace MM
{
    NN::NN(std::vector<double> inputs, int h1size, int outsize)
    {
        srand(0);
        for(std::vector<double>::iterator ptr = inputs.begin();ptr != inputs.end();ptr++)
        {
            input.push_back(*ptr);
        }
        for(int i = 0;i<h1size;i++)
        {
            h1.push_back(0.0);
        }
        for(int i = 0;i<outsize;i++)
        {
            out.push_back(0.0);
        }
        for(int i = 0;i<input.size()*h1size;i++)
        {
            wi.push_back((rand() % 20 - 10) / 10.0);
        }
        
    }

    double NN::relu(double d)
    {
        if(d > 0)
        {
            return d;
        }
        return 0;
    }

    void NN::fprop()
    {
        //Multiply the input layer with the weights between the input and h1 layers
        std::vector<double> inputwi = mmultiply(input, wi);
        //Fill the hidden layer with the products of the matrix multiplication put through the ReLU-function
        for(int i = 0;i<h1.size();i++)
        {
            h1[i] = relu(inputh1[i]);
        }
    }

    std::vector<double> NN::mmultiply(std::vector<double> left, std::vector<double> right)
    {
        int lsize = left.size();
        int rsize = right.size();
        int osize = rsize/lsize;
        std::vector<double> output(osize);
        for(int i = 0;i < lsize;i++)
        {
            for(int j = 0;j < osize;j++)
            {
                output[j] += left[i] * right[i*osize + j];
            }
        }
        return output;
    }
}