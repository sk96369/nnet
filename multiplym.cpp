#include "multiplym.h"
#include <vector>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <list>
#include "zlib.h"
#include <errno.h>
#include <stdio.h>
#include <tuple>

/* The following macro calls a zlib routine and checks the return
   value. If the return value ("status") is not OK, it prints an error
   message and exits the program. Zlib's error statuses are all less
   than zero. */

#define CALL_ZLIB(x) {                                                  \
        int status;                                                     \
        status = x;                                                     \
        if (status < 0) {                                               \
            fprintf (stderr,                                            \
                     "%s:%d: %s returned a bad status of %d.\n",        \
                     __FILE__, __LINE__, #x, status);                   \
            exit (EXIT_FAILURE);                                        \
        }                                                               \
    }

/* if "test" is true, print an error message and halt execution. */

#define FAIL(test,message) {                             \
        if (test) {                                      \
            inflateEnd (& strm);                         \
            fprintf (stderr, "%s:%d: " message           \
                     " file '%s' failed: %s\n",          \
                     __FILE__, __LINE__, file_name,      \
                     strerror (errno));                  \
            exit (EXIT_FAILURE);                         \
        }                                                \
    }

#define windowBits 15
#define ENABLE_ZLIB_GZIP 32
#define CHUNK 0x4000

//multiplym.cpp

void softmax(std::vector<double> &in, std::size_t size)
{
    assert(0 <= size <= sizeof(in) / sizeof(double));
    int sum = 0;
    for(int i = 0;i<size;++i)
    {
        sum += std::exp(in[i]);
    }
    for(int i = 0;i<size;++i)
    {
        in[i] = std::exp(in[i]/sum);
    }
}

namespace MM
{
    NN::NN(std::vector<double> inputs, int h1size, int outsize) : modelNr(count)
    {
        count++;
         
        srand(0);
        for(std::vector<double>::iterator ptr = inputs.begin();ptr != inputs.end();ptr++)
        {
            input.push_back(*ptr);
        }
        for(int i = 0;i<h1size;i++)
        {
            bias1.push_back((rand() % 20 - 10) / 10.0);
            h1.push_back(0.0);
        }
        for(int i = 0;i<outsize;i++)
        {
            bias2.push_back((rand() % 20 - 10) / 10.0);
            out.push_back(0.0);
        }
        for(int i = 0;i<input.size()*h1size;i++)
        {
            wi.push_back((rand() % 20 - 10) / 10.0);
        }

        learningrate = 0.1;
    }

    bool NN::saveModel()
    {
        return 1;
    }

    double NN::relu(double d) const
    {
        if(d > 0)
        {
            return d;
        }
        return 0;
    }

    std::vector<double> NN::drelu(const std::vector<double> &vec) const
    {
        std::vector<double> d;
        for(int i = 0;i<vec.size();i++)
        {
            //This works because the slope past 0 is 1, and 0 at 0 and below
            d.push_back(vec[i]>0);
        }
        return d;
    }

    void NN::fprop(const std::vector<double> &in)
    {
        //Multiply the input layer with the weights between the input and h1 layers
        std::vector<double> inputwi = lmultiply(input, wi);
        //Run the outputs of the matrix multiplication through the ReLU function to get the hidden states
        for(int i = 0;i<h1.size();i++)
        {
            h1[i] = relu(inputwi[i]);
        }
        //Run the outputs of the matrix multiplication through the ReLU function to get the final outputs
        std::vector<double> h1w = lmultiply(h1, w1);
        for(int i = 0;i<h1w.size();i++)
        {
            out[i] = relu(h1w[i]);
        }
        //Run the output layer through the softmax function
        softmax(out, out.size());
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
        //The variables holding the info of how much the weights should be adjusted
        std::vector<double> dh1outw;
        std::vector<double> dih1w;
        //The variables holding the info of how much the biases should be adjusted
        double dbias1;
        double dbias2;
        //Variable for the difference between target and generated output
        std::vector<double> diff1;
        std::vector<double> diff2;

        //Calculate the difference between target and generated output
        for(int i = 0;i<10;i++)
        {
            diff2.push_back(out[i]-targetoutput[i]);
        }

        //Calculate the adjustments needed for the weights and biases of the second layer
        dh1outw = wmultiply(h1, diff2);
        dbias2 = bsum(diff2);

        //Calculate the adjustments needed for the weights and biases of the first layer
        diff1 = lmultiply(drelu(h1), w1);                                                           //CHECK THE DIRECTION OF THIS IF TRAINING DOESNT WORK
        dih1w = wmultiply(input, diff1);
        dbias1 = bsum(diff1);
        
    }

    void NN::updateParameters(const std::vector<double> &dih1w, const std::vector<double> &dh1outw, double dbias1, double dbias2)
    {
        for(int i = 0;i<dih1w.size();i++)
        {
            wi[i] = wi[i] - learningrate * dih1w[i];
        }
        for(int i = 0;i<dh1outw.size();i++)
        {
            w1[i] = wi[i] - learningrate * dh1outw[i];
        }
        for(int i = 0;i<bias1.size();i++)
        {
            bias1[i] = bias1[i] - learningrate * dbias1;
        }
        for(int i = 0;i<bias2.size();i++)
        {
            bias2[i] = bias2[i] - learningrate * dbias2;
        }
    }

    void NN::train(const std::list<std::tuple<std::vector<double>, std::vector<double>>> &trainingdata)
    {
        for(std::list<std::tuple<std::vector<double>, std::vector<double>>>::const_iterator ptr = trainingdata.begin();ptr != trainingdata.end();ptr++)
        {
            fprop(std::get<0>(*ptr));
            bprop(std::get<1>(*ptr));
        }
    }

    std::vector<double> NN::lmultiply(const std::vector<double> &left, const std::vector<double> &right) const
    {
        int lsize = left.size();
        int rsize = right.size();
        //osize is the number of "columns" in the weight matrix, sort of
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

    std::vector<double> NN::wmultiply(const std::vector<double> &left, const std::vector<double> &right) const
    {
        std::vector<double> weights;
        for(int i = 0;i<left.size();i++)
        {
            for(int j = 0;j<right.size();j++)
            {
                weights[i*10+j] = left[i] * right[j];
            }
        }
        return weights;
    }

    double NN::bsum(const std::vector<double> &v) const
    {
        double d;
        for(int i = 0;i<v.size();i++)
        {
            d += v[i];
        }
        return d;
    }
}