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
			if(oh[i][j] > oh[i][max])
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
	void nnet::setParameters(const std::string filename)
	{
		std::ifstream file;
		file.open(filename);
		if(file.is_open())
		{
			double val;
			for(int i = 0;i<bias1.columns();i++)
			{	
				file >> val;
				bias1[0][i] = val;
			}
			for(int i = 0;i<bias2.columns();i++)
			{
				file >> val;
				bias2[0][i] = val;
			}
			for(int i = 0;i<wi.rows();i++)
			{
				for(int j = 0;j<wi.columns();j++)
				{
					file >> val;
					wi[i][j] = val;
				}
			}
			for(int i = 0;i<w1.rows();i++)
			{
				for(int j = 0;j<w1.columns();j++)
				{
					file >> val;
					w1[i][j] = val;
				}
			}
			file.close();
		}
		else
			std::cout << "File not found!\n";
	}	

	nnet::nnet(const mat<int> &in, int h1size, int outsize, int batch_size) : input(getTranspose(in)),  h1(0.0, batch_size, h1size), relu_h1(0.0, batch_size, h1size), bias1(0.01, 10, 1), bias2(0.01, 10, 1), out(0.0, batch_size, outsize), h2(0.0, batch_size, outsize), wi(-0.5, 0.5, 784, 10), w1(-0.5, 0.5, 10, 10), learningrate(0.9)
	{}

	nnet::nnet()
	{
	}

	void nnet::fprop()
	{
//std::cout << "Input matrix: [" << input.columns() << "][" << input.rows() << "]\n";
		//Multiply the input layer with the weights between the input and h1 layers
		h1.newValues(mm(wi, input));
//std::cout << "First hidden matrix: [" << h1.columns() << "][" << h1.rows() << "]\n";
		//Add the bias to each of the hidden states	
		add(h1, bias1);
		//Run relu function on the hidden states
		relu_h1.newValues(getRelu(h1));
//std::cout << "Hidden matrix after relu: [" << relu_h1.columns() << "][" << relu_h1.rows() << "]\n";
		h2.newValues(mm(w1, relu_h1));
//std::cout << "Second hidden matrix: [" << h2.columns() << "][" << h2.rows() << "]\n";
		add(h2, bias2);
		mat<double> h2_transposed(getTranspose(h2));
//std::cout << "Second hidden matrix after addition and transpose: [" << h2.columns() << "][" << h2.rows() << "]\n";
		out.newValues(getSoftmax(h2_transposed));
		out.transpose();
//std::cout << "Input matrix: " << input.toString() << std::cin.get() << std::endl;
//std::cout << "Non-softmaxed output matrix: " << h2_transposed.toString() << std::cin.get() << std::endl;
//std::cout << "Output matrix: " << out.toString() << std::cin.get() << std::endl;
//std::cout << "Output matrix after: [" << out.columns() << "][" << out.rows() << "]\n";
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
		mat<int> transposed_targetoutput(getTranspose(targetoutput));
		//Calculate the difference between target and generated output
		mat<double>delta(getError(transposed_targetoutput, out));
//			std::cout << "Output: " << out.toString() << std::cin.get() << std::endl;
//			std::cout << "Output error: " << delta.toString() << std::cin.get() << std::endl;
		//Calculate the adjustments needed for the weights and biases of the second layer
		mat<double> d_w1(scalar_m(mm(delta, getTranspose(relu_h1)), 1.0/(double)batch_size));
		mat<double>dbias2(scalar_m(sum_m(delta), 1.0/(double)batch_size));
		//Calculate the adjustments needed for the weights and biases of the first layer
		mat<double>delta2(hadamard(mm(getTranspose(w1),delta), drelu(h1))); 
//std::cout << "Second error matrix: " << delta2.toString() << std::cin.get() << std::endl;
		mat<double>d_inputweights(scalar_m(mm(delta2, getTranspose(input)), 1.0/(double)batch_size));
		mat<double>dbias1(scalar_m(sum_m(delta), 1.0/(double)batch_size));
		d_inputweights.transpose();
		d_w1.transpose();
		updateParameters(d_inputweights, d_w1, dbias1, dbias2); 
	}
	
	void nnet::updateParameters(const mat<double> &d_inputweights, const mat<double> &d_w1, mat<double> dbias1, mat<double> dbias2)
	{
//		std::cout << "Input weights before: " << d_inputweights.toString();
//		std::cin.get();
		for(int i = 0;i<wi.rows();i++)
		{
			for(int j = 0;j<wi.columns();j++)
			{
				wi[i][j] = wi[i][j] - learningrate * d_inputweights[i][j];
			}
		}
//		std::cout << "\nInput weights after: " << wi.toString();
//		std::cin.get();
//		std::cout << "Second weights before: " << d_w1.toString();
//		std::cin.get();
		for(int i = 0;i<w1.rows();i++)
		{
			for(int j = 0;j<w1.columns();j++)
			{
				w1[i][j] = w1[i][j] - learningrate * d_w1[i][j];
			}
		}
//		std::cout << "Second weights after: " << w1.toString();
//		std::cin.get();
//		std::cout << "Bias1 before: " << dbias1.toString();
//		std::cin.get();
		for(int i = 0;i<bias1.columns();i++)
		{
			bias1[0][i] -= learningrate * dbias1[0][i];
		}

//		std::cout << "\nBias1 after: " << bias1.toString();
//		std::cin.get();
//		std::cout << "Bias2 before: " << dbias2.toString();
//		std::cin.get();
		for(int i = 0;i<bias2.columns();i++)
		{
			bias2[0][i] -= learningrate * dbias2[0][i];
		}
//		std::cout << "\nBias2 after: " << bias2.toString();
//		std::cin.get();
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
			std::cout << "Prediction accuracy: " << accuracy*100 << "%\n";

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
		fprop();
	}
	
	std::vector<int> nnet::getPredictions() const
	{
		std::vector<int> predictions = onehot_toInt(out);
		return predictions;
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
			input[j][k] = vec[i];
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
				if(input[i][j] == 0)
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
