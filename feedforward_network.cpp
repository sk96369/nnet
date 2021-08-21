#include <time.h>
#include <cstdlib>
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
	nnet::nnet(const std::vector<int> &dimensions, int f_max, double learningrate) : features_maxvalue(f_max), learningrate(learningrate), size(dimensions.size() - 1), hidden_layers(dimensions.size() - 1), weights(dimensions.size() - 1), input(dimensions[0]), input_normalized(dimensions[0]), outputs(dimensions.size() - 1), biases(dimensions.size() - 1)
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

	nnet::nnet() : features_maxvalue(0), learningrate(0), size(0), hidden_layers(0), weights(0), input(0), input_normalized(0), outputs(0), biases(0)
	{
	}
	
	void nnet::trainRandom(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int datasize, int epoch)
	{
		
		std::vector<int> imagebatch;
		std::vector<int> labelbatch;
		std::vector<int> all_indexes(datasize);
		for(int i = 0;i<datasize;i++)
			all_indexes[i] = i;
		srand(time(NULL));
		std::cout << "Starting training with batch size: " << batchsize << ", epochs: " << epoch << "\n";
		for(int i = 0;i<epoch;i++)
		{
			std::vector<int> all_indexes_hat = all_indexes;
			std::vector<int> indexes_randomized;
			std::vector<int> imagebag;
			std::vector<int> labelbag;
			int upperlimit;
			int randomindex;
			for(int j = 0;j<datasize;j++)
			{
				upperlimit = datasize - j;
				randomindex = rand() % upperlimit;
				indexes_randomized.push_back(all_indexes_hat[randomindex]);
				all_indexes_hat.erase(all_indexes_hat.begin() + randomindex);
			}
			for(auto& j : indexes_randomized)
			{
				labelbag.push_back(labels[j]);
				for(int k = 0;k<imagesize;k++)
				{
					imagebag.push_back(j*imagesize + k);
				}
			}
			auto image_start = imagebag.begin();
			auto image_end = imagebag.begin() + batchsize*imagesize;
			auto label_start = labelbag.begin();
			auto label_end = labelbag.begin() + batchsize;

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
				std::cout << "Epoch: " << i << "/" << epoch << " - Iteration: " << j << "/" << iterations << std::endl;
			}
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
			std::cout << "Epoch: " << i << "/" << epoch << " - Iteration: " << j << "/" << iterations << std::endl;
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
		if(i < size)
			return hidden_layers[i];
		else
			return input_normalized;
	}

	const mat<double>& nnet::getLayer(int i) const
	{
		if(i == -1)
			return input_normalized;
		if(i < size)
			return hidden_layers[i];
		else
			return input_normalized;
	}

	mat<double>& nnet::getOutput(int i)
	{
		if(i == -1)
			return input_normalized;
		if(i < size)
			return outputs[i];
		else
			return input_normalized;
	}

	const mat<double>& nnet::getOutput(int i) const
	{
		if(i == -1)
			return input_normalized;
		if(i < size)
			return outputs[i];
		else
			return input_normalized;
	}

	int nnet::saveModel(std::string filename)
	{
		filename.append(".txt");
		std::ofstream file;
		file.open(filename);
		if(file.is_open())
		{
			file << size << "/";
			file << input.size() << " ";
			for(int i = 0;i<size;i++)
			{
				file << hidden_layers[i].size() << " ";
			}
			file << "/" << learningrate << " " << features_maxvalue << "/" ;

			for(int i = 0;i<size;i++)
			{
				file <<  biases[i].toString(15) << "/" << weights[i].toString(15) << "/";
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
		double parameter;
		int dimension;
		std::vector<int> dimensions;
		file.open(filename);
		if(file.is_open())
		{
			std::stringstream ss;
			std::getline(file, line, '/').good();
			std::stringstream ss_size;
			ss_size.str(line);
			ss_size >> size;
			std::getline(file, line, '/').good();
			ss.str(line);
			for(int i = 0;i<=size;i++)
			{
				ss >> dimension;
				dimensions.push_back(dimension);
			}
			std::getline(file, line, '/').good();
			std::stringstream ss_hyperparameters;
			ss_hyperparameters.str(line);
			ss_hyperparameters >> learningrate >> features_maxvalue;
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
				if(std::getline(file, line, '/').good())
				{
					std::stringstream parameter_ss;
					parameter_ss.str(line);
					std::vector<double> layer_of_parameters;
					while(parameter_ss.rdbuf()->in_avail())
					{
						parameter_ss >> parameter;
						layer_of_parameters.push_back(parameter);
					}
					mat<double> matrix_of_biases(layer_of_parameters, 1, dimensions[i+1]);
					biases.push_back(matrix_of_biases);

				}
			
				if(std::getline(file, line, '/').good())
				{
					std::stringstream parameter_ss;
					parameter_ss.str(line);
					std::vector<double> layer_of_parameters;
					while(parameter_ss.rdbuf()->in_avail())
					{
						parameter_ss >> parameter;
						layer_of_parameters.push_back(parameter);
					}
					mat<double> matrix_of_weights(layer_of_parameters, dimensions[i], dimensions[i+1]);
					weights.push_back(matrix_of_weights);
				}
			}
		}
		file.close();
	}
		
	void nnet::printLayer(int i, std::ostream &o, int precision)
	{
		o << getOutput(i).toString(precision);
	}


}
