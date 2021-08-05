#include <vector>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <tuple>
#include <float.h>
#include "onehot.h"
#include "multiplym.h"
#include "mm_math.h"
#define IMAGEWIDTH 28

/*
std::vector<int> onehot_toInt(const MM::mat<double> &oh)
{
	int columns = oh.columns();
	int rows = oh.rows();
	std::vector<int> outputs_as_integers;
	for(int i = 0;i<rows;i++)
	{
		int max = 0;
		for(int j = 0;j<columns;j++)
		{
			if(oh.m[i][j] > oh.m[i][max])
			{
				max = j;
			}
		}
		outputs_as_integers.push_back(max);
	}
	return outputs_as_integers;
}
*/

//multiplym.cpp
namespace MM
{
	
/*
	nnet::nnet(int inputsize, int h1size, int outsize, std::string filename)
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

	nnet::nnet(const mat<int> &in, int h1size, int outsize, int batch_size) : input(in),  h1(0.0, batch_size, h1size), relu_h1(0.0, batch_size, h1size), bias1(0.01, 10, 1), bias2(0.01, 10, 1), out(0.0, outsize, batch_size), h2(0.0, outsize, batch_size), wi(-0.5, 0.5, 784, 10), w1(-0.5, 0.5, 10, 10), learningrate(0.9)
	{}


	void nnet::fprop()
	{
		//Multiply the input layer with the weights between the input and h1 layers
		mat<int> transposed = getTranspose(input);
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

	double calculate_prediction(std::vector<int> labels, std::vector<int> predictions)
	{
		int wrong = 0;
		int correct = 0;
		for(int i = 0;i < labels.size();i++)
		{
			if(labels[i] == predictions[i])
			{
				correct++;
			}
			else
				wrong++;
		}
		return (double)correct / labels.size();
	}
	
	void nnet::bprop(const mat<int> &targetoutput)
	{
		int batch_size = targetoutput.rows();
		
		//Calculate the difference between target and generated output
		std::cout << "TEST\n";	
		mat<double>delta = getError(targetoutput, out);
		//Calculate the adjustments needed for the weights and biases of the second layer
		mat<double> d_w1 = scalar_m(mm(h1, delta), 1/batch_size);
		mat<double>dbias2 = scalar_m(sum_m(delta), 1/batch_size);
		//Calculate the adjustments needed for the weights and biases of the first layer
		mat<double>delta2 = hadamard(mm(w1,getTranspose(delta)), drelu(h1)); 
		mat<double>d_inputweights = scalar_m(mm(delta2, input), 1/batch_size);
		mat<double>dbias1 = scalar_m(sum_m(delta), 1/batch_size);
		updateParameters(d_inputweights, d_w1, dbias1, dbias2); 
	}
	
	void nnet::updateParameters(const mat<double> &d_inputweights, const mat<double> &d_w1, mat<double> dbias1, mat<double> dbias2)
	{
		for(int i = 0;i<wi.rows();i++)
		{
			for(int j = 0;j<wi.columns();j++)
			{
				wi.m[i][j] = wi.m[i][j] - learningrate * d_inputweights.m[i][j];
			}
		}
		for(int i = 0;i<w1.rows();i++)
		{
			for(int j = 0;j<w1.columns();j++)
			{
				w1.m[i][j] = w1.m[i][j] - learningrate * d_w1.m[i][j];
			}
		}
		for(int i = 0;i<bias1.rows();i++)
		{
			bias1.m[0][i] -= learningrate * dbias1.m[0][i];
		}

		for(int i = 0;i<bias2.rows();i++)
		{
			bias2.m[0][i] -= learningrate * dbias2.m[0][i];
		}
	}


        void nnet::train(std::vector<int> labels, int batch_size, int epoch)
	{
		int correct = 0;
		int wrong = 0;
		size_t interval = 0;
		size_t count = 0;
		for(int j = 0;j < epoch;j++)
		{
        		fprop();

			double accuracy = calculate_prediction(labels, onehot_toInt(out));
			std::cout << "Prediction accuracy: " << accuracy/100 << "%\n";

			mat<int> oh_labels = int_toOneHot(labels, 10);
			bprop(oh_labels);
		}
//			std::cin.get();
	}

/*	bool nnet::setInput(const std::vector<int> &in)
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
	bool nnet::saveModel(std::string filename) const
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

	void nnet::predict(const mat<int> &in)
	{
		input = in;
		std::cout << "TEST2\n";
		fprop();
	}

	void nnet::setInput(const mat<int> &in)
	{
		mat<int> newinput(in);
		/*for(auto& i : in.m)
		{
			for(auto& j : i)
			{
				std::cout << " " << j;
			}
			std::cout << std::endl;
		}
		for(auto& i : newinput.m)
		{
			for(auto& j : i)
			{
				std::cout << " " << j;
			}
			std::cout << std::endl;
		}*/
		input = newinput;
	}

	void nnet::setInput(const std::vector<int> &vec, int x, int y)
	{
		int j = 0;
		int k = 0;
		for(int i = 0;i < vec.size();i++)
		{
			input.m[j][k] = vec[i];
			k++;
			if(k == x)
			{
				k = 0;
				j++;
			}
		}
	}

	std::string nnet::toString()
	{
		std::string str = "";
		std::string test = "";
		std::vector<int> intlist = onehot_toInt(out);
		for(int i = 0;i<out.rows();i++)
		{
			int k = 0;
			for(int j = 0;j<input.columns();j++)
			{
				if(input.m[i][j] == 0)
					str.append(" ");
				else
					str.append("#");
				k++;
				if(k == IMAGEWIDTH)
				{
					k = 0;
					str.append("\n");
				}
			}
			str += "\nPredicted output: " + std::to_string(intlist[i]) + "\n";
		}
		return str;
	}
}
