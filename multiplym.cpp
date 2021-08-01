#include "multiplym.h"
#include <sstream>
#include <vector>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <list>
#include <errno.h>
#include <stdio.h>
#include <tuple>
#include "onehot.cpp"
#include <float.h>
#include "mm_math.h"

//multiplym.cpp
namespace MM
{
	
/*
	NN::NN(int inputsize, int h1size, int outsize, std::string filename)
	{
		
		for(int i = 0;i<inputsize;i++)
		{
			input.push_back(0);
		}
		for(int i = 0;i<h1size;i++)
		{
			h1.push_back(0);
		}
		for(int i = 0;i<outsize;i++)
		{
			out.push_back(0);
		}

		std::ifstream file;
		file.open(filename);
		if(file)
		{
			size_t phase = 0;
			double val;
			while(phase < 10)
			{
				file >> val;
				std::cout << val << "\n";
				bias1.push_back(val);
				phase++;
			}
			while(phase < 20)
			{
				file >> val;
				bias2.push_back(val);
				phase++;
			}
			while(phase < 7840 + 20)
			{
				file >> val;
				wi.push_back(val);
				phase++;
			}
			while(file >> val)
			{
				w1.push_back(val);
				phase++;
			}
			file.close();
		}
		else
			std::cout << "File not found!\n";
	}			

*/

	template<typename A>
	NN::NN(mat<A> in, int h1size, int outsize, int batch_size) 
	{
	
		input = in;
		h1 = mat<double>(0.0, batch_size, h1size);
		relu_h1 = mat<double>(h1);
		bias1 = mat<double>(0.01, 10, 1);
		bias2 = mat<double>(0.01, 10, 1);
		out = mat<double>(0.0, outsize, batch_size);
		h2 = mat<double>(out);
		wi = mat<double>(-0.5, 0.5, 10, 10);
		w1 = mat<double>(-0.5, 0.5, 10, 10);
        	learningrate = 0.08;
	}

    

    void NN::fprop()
    {
        //Multiply the input layer with the weights between the input and h1 layers
	mat<int> transposed = input.transpose();
	h1 = mm(wi, transposed);
	//Add the bias to each of the hidden states	
	add(h1, bias1);
        //Run relu function on the hidden states
	relu_h1 =getRelu(h1);
	h2 = mm(w1, relu_h1);
	add(h2, bias2);
	h2.transpose();
        out = getSoftmax(h2);
    }

    void NN::bprop(const mat<int> &targetoutput)
    {
	int batch_size = targetoutput.rows();

        //Calculate the difference between target and generated output
	mat<double>delta = getError(targetoutput, out);
        //Calculate the adjustments needed for the weights and biases of the second layer
	mat<double> d_w1 = scalar_m(mm(h1, delta), 1/batch_size);
	mat<double>dbias2 = scalar_m(sum_m(delta), 1/batch_size);

        //Calculate the adjustments needed for the weights and biases of the first layer
	mat<double>delta2 = hadamard(mm(w1,getTranspose(delta)), drelu(h1)); 

        mat<double>d_inputweights = scalar_m(mm(input, getTranspose(delta2)), 1/batch_size);

        mat<double>dbias1 = scalar_m(sum_m(delta), 1/batch_size);

       	updateParameters(d_inputweights, d_w1, dbias1, dbias2); 
    }

    void NN::updateParameters(const mat<double> &d_inputweights, const mat<double> &d_w1, mat<double> dbias1, mat<double> dbias2)
    {
        for(int i = 0;i<wi.columns();i++)
        {
		for(int j = 0;j<wi.rows();j++)
		{
			wi.m[i][j] = wi.m[i][j] - learningrate * d_inputweights.m[i][j];
		}
        }
        for(int i = 0;i<w1.columns();i++)
        {
		for(int j = 0;j<w1.rows();j++)
		{
			w1.m[i][j] = w1.m[i][j] - learningrate * d_w1.m[i][j];
		}
        }
        for(int i = 0;i<bias1.columns();i++)
        {
		bias1.m[0][i] -= learningrate * dbias1.m[0][i];
        }
	
	for(int i = 0;i<bias2.columns();i++)
        {
		bias2.m[0][i] -= learningrate * dbias2.m[0][i];
        }
    }


        void NN::train(std::vector<int> labels, int batch_size, int epoch)
	{
		int correct = 0;
		int wrong = 0;
		size_t interval = 0;
		size_t count = 0;
		for(int j = 0;j < epoch;j++)
		{
        		fprop();
			mat<int> oh_labels = int_toOneHot(labels, 10);

			bprop(oh_labels);
		}
//			std::cin.get();
	}

/*	bool NN::setInput(const std::vector<int> &in)
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
*/
    bool NN::saveModel(std::string filename) const
    {
	    filename.append(".txt");
	std::ofstream file;
	file.open(filename);
	for(auto& i : bias1.m)
	{
		for(auto& j : i)
			file << j << " ";
	}
	file << std::endl;

	for(auto& i : bias2.m)
	{
		for(auto& j : i)
			file << j << " ";
	}
	file << std::endl;

	for(auto& i : wi.m)
	{
		for(auto& j : i)
			file << j << " ";
	}

	for(auto& i : w1.m)
	{
		for(auto& j : i)
			file << j << " ";
	}
	file << std::endl;

	file.close();
	return 1;
    }

    std::vector<int> NN::predict(const mat<int> &in)
    {
	    input = in;
	    std::cout << "TEST2\n";
	    fprop();
	    std::cout << "TEST3\n";
	    return onehot_toInt(out);
    }


}
