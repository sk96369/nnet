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

    MM::NN network(inputs, 10, 10);

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
        bias1 = 0.2;
        bias2 = 0.2;
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
        //Run the outputs of the matrix multiplication through the ReLU function to get the hidden states
        for(int i = 0;i<h1.size();i++)
        {
            h1[i] = relu(inputwi[i]);
        }
        //Run the outputs of the matrix multiplication through the ReLU function to get the final outputs
        std::vector<double> h1w = mmultiply(h1, w1);
        for(int i = 0;i<h1w.size();i++)
        {
            out[i] = relu(h1w[i]);
        }
    }

    std::vector<double> NN::toOneHot(int v)
    {
        std::vector<double> vv;
        for(int i = 0;i<10;i++)
        {
            if(i == v)
            {
                vv.push_back(1);
            }
            else
            {
                vv.push_back(0);
            }
        }
        return vv;
    }

    void NN::bprop(const std::vector<double> targetoutput)
    {
        
    }

    void NN::train()
    {

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