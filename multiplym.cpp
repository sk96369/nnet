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
#include "onehot.cpp"
#include <float.h>

//multiplym.cpp
namespace MM
{
	void softmax(std::vector<double> &in, std::size_t size)
	{
    		double sum = 0;
    		for(size_t i = 0;i<size;++i)
    		{
			double j = std::exp(in[i]);
			sum += j;
    		}
    		for(size_t i = 0;i<size;++i)
    		{
    		    in[i] = std::exp(in[i])/sum;
    		}
		//Check for a nan-value, and change it to 1 if one is found
		for(auto& i : in)
		{
			if(isnan(i))
				i = 1;
		}
	}

    NN::NN(int h1size, int outsize) 
    {
        srand(0);
        for(int i = 0;i<28*28;i++)
        {
            input.push_back(0);
        }
        for(int i = 0;i<h1size;i++)
        {
            bias1.push_back((rand() % 20 - (double)10) / 15.0);
            h1.push_back(0.0);
        }
        for(int i = 0;i<outsize;i++)
        {
            bias2.push_back((rand() % 20 - (double)5) / 15.0);
            out.push_back(0.0);
        }
	int wisize = input.size()*h1size;
        for(int i = 0;i<wisize;i++)
        {
            wi.push_back((rand() % 10 - (double)5) / 15.0);
        }
	int w1size = 100;
	for(int i = 0;i<w1size;i++)
	{
		w1.push_back((rand() % 10 - (double)5) / 15.0);
	}
        learningrate = 0.1;
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



    void NN::fprop(const std::vector<int> &in)
    {
        //Multiply the input layer with the weights between the input and h1 layers
        std::vector<double> inputwi = lmultiply(in, wi);
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
	//	std::cout << h1w[i] << " - " << out[i] << std::endl;
        }
        //Run the output layer through the softmax function
        softmax(out, out.size());
	//for(auto& i : out)
	//	std::cout << i << " ";
	//std::cout << std::endl;
	//std::cout << out[4] << std::endl;
	//std::cout << onehot_toInt(out) << std::endl;
    }

    void NN::fprop()
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

    void NN::bprop(const std::vector<int> targetoutput)
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

//	std::cout << h1.size() << " - " << diff2.size() << std::endl;
        //Calculate the adjustments needed for the weights and biases of the second layer
        dh1outw = wmultiply(h1, diff2);

//	    std::cout << "wmultiply success!" << std::endl;
        dbias2 = bsum(diff2);
//	std::cout << "bsum success!" << std::endl;

        //Calculate the adjustments needed for the weights and biases of the first layer
        diff1 = lmultiply(drelu(h1), w1); 
//	std::cout << "lmultiply success!" << std::endl;
	//CHECK THE DIRECTION OF THIS IF TRAINING DOESNT WORK
        dih1w = wmultiply(input, diff1);
//	std::cout << "wmultiply2 success!" << std::endl;
        dbias1 = bsum(diff1);
//	std::cout << "bsum success!" << std::endl;
       	updateParameters(dih1w, dh1outw, dbias1, dbias2); 
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

	void NN::train(const std::list<std::tuple<std::vector<int>, int>> &trainingdata, unsigned int iterations)
	{
		int correct = 0;
		int wrong = 0;
		for(auto ptr = trainingdata.begin();ptr != trainingdata.end();ptr++)
        	{
			for(int i = 0;i < iterations;i++)
			{
        			fprop(std::get<0>(*ptr));
				int int_output = onehot_toInt(out);
				int label = std::get<1>(*ptr);
	//			for(auto& i : out)
	//				std::cout << i << " ";
	//			std::cout << std::endl;
	//			std::cout << "Output: " << int_output << " - Reference output: " <<  label << std::endl;
            			bprop(int_toOneHot(std::get<1>(*ptr), 10));
				if(int_output != label)
					wrong++;
				else
					correct++;
			}
			std::cout<< "Accuracy: " << ((double)correct/((double)correct+(double)wrong))*100 << "%\n";
	        }
	}

    std::vector<double> NN::lmultiply(const std::vector<int> &left, const std::vector<double> &right) const
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
                output[j] += (double)left[i] * right[i*osize + j];
            }
        }
        return output;
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
        std::vector<double> weights(left.size()*right.size(), 0);
        for(int i = 0;i<left.size();i++)
        {
            for(int j = 0;j<right.size();j++)
            {
                weights[i*10+j] = left[i] * right[j];
            }
        }
        return weights;
    }

    std::vector<double> NN::wmultiply(const std::vector<int> &left, const std::vector<double> &right) const
    {
        
        std::vector<double> weights(left.size()*right.size(), 0);
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
	
    bool NN::setInput(const std::vector<int> &in)
    {
	    if(in.size() == 28*28)
	    {
		    for(int i = 0;i < 28*28;i++)
		    {
			    input[i] = in[i];
		    }
	    }
	    else
		    return false;
	    return true;
    }

    bool NN::saveModel(std::string filename) const
    {
	    filename.append(".txt");
	std::ofstream file;
	file.open(filename);
	for(auto ptr = bias1.begin();ptr != bias1.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = bias2.begin();ptr != bias2.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = wi.begin();ptr != wi.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = w1.begin();ptr != w1.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;
	file.close();
	return 1;
    }

    int NN::predict(const std::vector<int> &in)
    {
	    input = in;
	    fprop();
	    return onehot_toInt(out);
    }


}
