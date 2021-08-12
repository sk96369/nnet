#include <vector>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <tuple>
#include <float.h>
#include "onehot.h"
#include "mm_math.h"
#include "feedforward_network.h"
#include <iterator>

namespace MM
{
	nnet::nnet(const std::vector<int> &dimensions, int f_max) : features_maxvalue(f_max), learningrate(0.1), size(dimensions.size() - 1), hidden_layers(dimensions.size() - 1), weights(dimensions.size() - 1), input(dimensions[0]), input_normalized(dimensions[0]), outputs(dimensions.size() - 1), biases(dimensions.size() - 1)
	{
		
		for(int i = 0;i<size;i++)
		{
			mat<double> newlayer(dimensions[i + 1]);
			mat<double> newoutputlayer(dimensions[i+1]);
			hidden_layers[i] = newlayer;
			outputs[i] = newoutputlayer;
		}
		for(int i = 0;i<size;i++)
		{
			mat<double> newweight(-0.5, 0.5, getLayer(i-1).size(), getLayer(i).size());
			weights[i] = newweight;
		}
		for(int i = 0;i<size;i++)
		{
			mat<double> newbias(-0.2, 0.2, 1, getLayer(i).rows());
			biases[i] = newbias;
		}
	}

	void nnet::train(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int imagewidth, int datasize, int epoch)
	{
		std::vector<int> imagebatch;
		std::vector<int> labelbatch;
		auto image_start = images.begin();
		auto image_end = images.begin() + batchsize*imagesize;
		auto label_start = labels.begin();
		auto label_end = labels.begin() + batchsize;

		int iterations = (epoch * datasize) / batchsize;
		for(int i = 0;i<iterations;i++)
		{
			imagebatch.assign(image_start, image_end);
			labelbatch.assign(label_start, label_end);
			//Propagate forward
			setInput(imagebatch);
			
			std::cout << "Fprop:\n";
			fprop();

			//Print outputs
			std::vector<int> vec = onehot_toInt(getOutput(1));
			for(int ptr = 0;ptr < vec.size();ptr++)
			{
				std::cout << vec[ptr] << " ";
			}
			std::cout << std::endl;

			//Backpropagate
			std::cout << "Bprop:\n";
			bprop(labelbatch);

			//Move the iterators
			image_start += batchsize*imagesize;
			label_start += batchsize;
			if(image_start != images.end())
			{
				image_end += batchsize*imagesize;
				label_end += batchsize;
			}
		}
	}

	void nnet::fprop()
	{
		int i = 0;
		for(;i<size-1;i++)
		{
			std::cout << "Layer " << i << " mm\n";
			getLayer(i).newValues(add(mm(weights[i], getLayer(i-1)), biases[i]));
			getOutput(i).newValues(getRelu(getLayer(i)));

		}
			std::cout << "Layer " << i << " mm\n";
		getLayer(i).newValues(add(mm(weights[i], getLayer(i-1)), biases[i]));
		getOutput(i).newValues(getSoftmax(getLayer(i)));
	}

	mat<double> nnet::getOutput()
	{
		return outputs[size-1];
	}

	void nnet::bprop(const std::vector<int> &labels)
	{
		int i = size-1;
		std::vector<mat<double>> weights_delta(size);
		std::vector<mat<double>> biases_delta(size);
		mat<double> target_output(int_toOneHot(labels, outputs[size-1].size()));

		mat<double> delta = getError(target_output, outputs[i]);

		std::cout << target_output.toString() << std::endl;
		std::cout << getOutput().toString() << std::endl;
		std::cout << delta.toString() << std::endl;

		weights_delta[i] = (mm(scalar_m(delta, 1/delta.columns()), getTranspose(outputs[i])));
		biases_delta[i] = (sum_m(scalar_m(delta, 1/delta.columns())));

//		printf("Hidden layer[%i]: %i %i - Outputs[%i]: %i %i\n", i, getLayer(i).columns(), getLayer(i).rows(), i, getOutput(i).columns(), getOutput(i).rows());
//		printf("Delta[%i]: %i %i - Weights[%i]: %i %i - Weightdelta[%i]: %i %i\n", i, delta.columns(), delta.rows(), i, weights[i].columns(), weights[i].rows(), i, weights_delta[i].columns(), weights_delta[i].rows());

		for(i--;i>=0;i--)
		{
			delta.newValues(hadamard(mm(getTranspose(weights[i+1]), delta), drelu(getLayer(i))));
			weights_delta[i] = mm(scalar_m(delta, 1/delta.columns()),
						getTranspose(getLayer(i-1)));
			biases_delta[i] = (sum_m(scalar_m(delta, 1/delta.columns())));

//		printf("Hidden layer[%i]: %i %i - Outputs[%i]: %i %i\n", i, getLayer(i).columns(), getLayer(i).rows(), i, getOutput(i).columns(), getOutput(i).rows());
//		printf("Delta[%i]: %i %i - Weights[%i]: %i %i - Weightdelta[%i]: %i %i\n", i, delta.columns(), delta.rows(), i, weights[i].columns(), weights[i].rows(), i, weights_delta[i].columns(), weights_delta[i].rows());

		}
		updateParameters(weights_delta, biases_delta);
	}

	void nnet::updateParameters(std::vector<mat<double>> weights_delta, std::vector<mat<double>> biases_delta)
	{
		for(int i = 0;i<size;i++)
		{
			weights[i].newValues(getError(weights_delta[i], scalar_m(weights_delta[i], learningrate)));
			biases[i].newValues(getError(biases_delta[i], scalar_m(biases_delta[i], learningrate)));
		}
	}

	std::vector<int> nnet::getDimensions()
	{
		std::vector<int> dimensions(size + 1);
		dimensions[0] = input.size();
		for(int i = 0;i<size;i++)
		{
			dimensions[i+1] = getLayer(i).size();
		}
		return dimensions;
	}

	void nnet::setInput(const std::vector<int> &newinput)
	{
		mat<int> newinput_mat(newinput, newinput.size()/input.rows(), input.rows());
		input_normalized.newValues(getNormalized(newinput_mat, features_maxvalue));
	}

	mat<double>& nnet::getLayer(int i)
	{
		if(i == -1)
			return input_normalized;
		return hidden_layers[i];
	}

	const mat<double>& nnet::getLayer(int i) const
	{
		if(i == -1)
			return input_normalized;
		return hidden_layers[i];
	}

	mat<double>& nnet::getOutput(int i)
	{
		if(i == -1)
			return input_normalized;
		return outputs[i];
	}

	const mat<double>& nnet::getOutput(int i) const
	{
		if(i == -1)
			return input_normalized;
		return outputs[i];
	}

	int nnet::saveModel(std::string filename)
	{
		filename.append(".txt");
		std::ofstream file;
		file.open(filename);
		if(file.is_open())
		{
			for(int i = 0;i<size;i++)
			{
				file << biases[i].toString() << weights[i].toString();
			}
		}
		file.close();
		return 0;
	}
}
