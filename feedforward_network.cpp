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
			mat<double> newweight(-0.5, 0.5, getLayer(i-1).rows(), getLayer(i).rows());
			weights[i] = newweight;
		}
		for(int i = 0;i<size;i++)
		{
			mat<double> newbias(-0.2, 0.2, 1, getLayer(i).rows());
			biases[i] = newbias;
		}
	}

	void nnet::train(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int datasize, int epoch)
	{
		std::vector<int> imagebatch;
		std::vector<int> labelbatch;
		std::cout << "Starting training with batch size: " << batchsize << ", epochs: " << epoch << "\n";
		for(int i = 0;i<epoch;i++)
		{
		auto image_start = images.begin();
		auto image_end = images.begin() + batchsize*imagesize;
		auto label_start = labels.begin();
		auto label_end = labels.begin() + batchsize;

		int iterations = datasize / batchsize;
		for(int j = 0;j<iterations;j++)
		{
			imagebatch.assign(image_start, image_end);
			labelbatch.assign(label_start, label_end);
			//Propagate forward
			setInput(imagebatch);
			fprop();

			//Backpropagate
			bprop(labelbatch);

			//Move the iterators
			image_start += batchsize*imagesize;
			label_start += batchsize;
			if(image_start != images.end())
			{
				image_end += batchsize*imagesize;
				label_end += batchsize;
			}
			std::cout << "Epoch: " << i << " - Iteration: " << j << std::endl;
		}
		}
	}

	void nnet::fprop()
	{
		int i = 0;
		for(;i<size-1;i++)
		{
			getLayer(i).newValues(add(mm(weights[i], getLayer(i-1)), biases[i]));
			getOutput(i).newValues(getRelu(getLayer(i)));

		}
		getLayer(i).newValues(add(mm(weights[i], getLayer(i-1)), biases[i]));
		getOutput(i).newValues(getSoftmax(getLayer(i)));
	}

	mat<double> nnet::getOutput()
	{
		return outputs[size-1];
	}

	void nnet::bprop(const std::vector<int> &labels)
	{
		std::vector<mat<double>> weights_delta(size);
		std::vector<mat<double>> biases_delta(size);
		mat<int> target_output(int_toOneHot(labels, getOutput().rows()));
		mat<double> delta = getError(outputs[size-1], target_output);
		double scalar = (double)1/(double)input_normalized.columns();

//		std::cout << "Weights at i = 0: " << weights[0].toString() << std::endl;
//		std::cout << "Weights at i = 1: " << weights[1].toString() << std::endl << std::endl;
//		std::cout << "Target output " << target_output.toString() << std::endl;
//		std::cout << "Prediction: " << getOutput().toString() << std::endl;
//		std::cout << "Delta[1]: " << delta.toString() << std::endl;

//		std::cout << "Biases delta[1]: " << biases_delta[i].toString() << std::endl << std::endl;
//		std::cout << "Weights delta[1]: " << weights_delta[i].toString() << std::endl << std::endl;
//		printf("Hidden layer[%i]: %i %i - Outputs[%i]: %i %i\n", i, getLayer(i).columns(), getLayer(i).rows(), i, getOutput(i).columns(), getOutput(i).rows());
//		printf("Delta[%i]: %i %i - Weights[%i]: %i %i - Weightdelta[%i]: %i %i\n", i, delta.columns(), delta.rows(), i, weights[i].columns(), weights[i].rows(), i, weights_delta[i].columns(), weights_delta[i].rows());

		for(int i = size-1;i>=0;i--)
		{
			if(i < size-1)
			{
				delta.newValues(hadamard(mm(getTranspose(weights[i+1]), delta),
							drelu(getLayer(i))));
			}
			weights_delta[i] = mm(scalar_m(delta, scalar),
						getTranspose(getOutput(i-1)));
			biases_delta[i] = sum_m(scalar_m(delta, scalar));
//			std::cout << std::endl << biases_delta[i].toString() << std::endl;


//		printf("Hidden layer[%i]: %i %i - Outputs[%i]: %i %i\n", i, getLayer(i).columns(), getLayer(i).rows(), i, getOutput(i).columns(), getOutput(i).rows());
//		printf("Delta[%i]: %i %i - Weights[%i]: %i %i - Weightdelta[%i]: %i %i\n", i, delta.columns(), delta.rows(), i, weights[i].columns(), weights[i].rows(), i, weights_delta[i].columns(), weights_delta[i].rows());

		}
//		std::cout << "Delta[0]: " << delta.toString() << std::endl << std::endl;
//		std::cout << "Weights delta[0]: " << weights_delta[0].toString() << std::endl << std::endl;
//		std::cin.get();
		updateParameters(weights_delta, biases_delta);
	}

	void nnet::updateParameters(std::vector<mat<double>> weights_delta, std::vector<mat<double>> biases_delta)
	{
		for(int i = 0;i<size;i++)
		{
			weights[i].newValues(getError(weights[i], scalar_m(weights_delta[i], learningrate)));
			biases[i].newValues(getError(biases[i], scalar_m(biases_delta[i], learningrate)));
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
				file << dimensions[i] << " ";
			}
			file << "/" << learningrate << " " << features_maxvalue << "/" ;

			for(int i = 0;i<size;i++)
			{
				file <<  biases[i].toString() << "/" weights[i].toString() << "/";
			}
		}
		file.close();
		return 0;
	}
		
	std::vector<int> nnet::predict(const std::vector<int> &data)
	{
		setInput(data);
		fprop();
		std::vector<int> predictions = onehot_toInt(getOutput());
		return predictions;
	}
		
	nnet::nnet(std::string filename)
	{
		filename.append(".txt");
		std::ifstream file;
		std::string line;
		std::stringstream ss;
		double parameter;
		int dimension;
		file.open(filename);
		if(file.is_open())
		{
			std::getline(file, line, '/').good();
			ss << line;
			while(ss.rdbuf()->in_avail())
			{
				ss >> dimension;
				dimensions.push_back(dimension);
			}
			size = dimensions.size()-1;
			std::getline(file, line, '/').good();
			ss << line;
			ss >> learningrate >> features_maxvalue;
			input = mat<int>(dimensions[0]);
			input_normalized = mat<double>(dimensions[0]);
			hidden_layers = std::vector<mat<double>>(size);
			outputs = std::vector<mat<double>>(size);
			for(int i = 0;i<size;i++)
			{
				hidden_layers[i] = mat<double>(dimensions[i+1]);
				outputs[i] = mat<double>(dimensions[i+1]);
			}
			for(int i = 0;i<size;i++)
			{
				std::getline(file, line, '/').good();
				ss << line;
				std::vector<double> layer_of_parameters;
				while(ss.rdbuf()->in_avail())
				{
					ss >> parameter;
					layer_of_parameters.push_back(parameter);
				}
				mat<double> matrix_of_biases(layer_of_parameters, 1, dimensions[i+1]);
				biases.push_back(matrix_of_biases);
				layer_of_parameters.clear();
				std::getline(file, line, '/').good();
				ss << line;
				while(ss.rdbuf()->in_avail())
				{
					ss >> parameter;
					layer_of_parameters.push_back(parameter);
				}
				mat<double> matrix_of_weights(layer_of_parameters, dimensions[i],
						dimensions[i+1]);
			}
		}
		file.close();
	}


}
