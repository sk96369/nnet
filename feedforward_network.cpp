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
#include <chrono> //For measuring time taken on functions

//Definitions

MM::mat<double> gpu_mm(const MM::mat<double> &left, const MM::mat<double> &right);

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
			mat<double> newbias(0.1, 0.3, 1, getLayer(i).rows());
			biases[i] = newbias;
		}
	}

	nnet::nnet() : features_maxvalue(0), learningrate(0), size(0), hidden_layers(0), weights(0), input(0), input_normalized(0), outputs(0), biases(0)
	{
	}
	
	void nnet::trainRandom(const std::vector<int> &images, const std::vector<int> &labels, int batchsize, int imagesize, int datasize, int epoch, bool printLabels, int imagewidth)
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
					imagebag.push_back(images[j*imagesize + k]);
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

				//For checking whether the training inputs match their labels
				if(j == 0 && printLabels && imagewidth > 0)
				{
					std::cout << "Input images:\n";
					printLayer(-1, std::cout, -1, imagewidth);
					std::cout << "\nLabels: \n";
					for(auto& label : labelbatch)
					{
						std::cout << label << std::endl;
					}
					//To pause until user presses a button.
					std::cout << "Press enter to continue.\n";
					std::cin.get();
				}

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
				std::cout << "Epoch: " << i+1 << "/" << epoch << " - Iteration: " << j+1 << "/" << iterations << "\r";
				fflush(stdout);
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
			std::cout << "gpu_mm - time taken: ";
			auto start = std::chrono::high_resolution_clock::now();

			mat<double> matrixproduct = gpu_mm(weights[i], getLayer(i-1));
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << duration.count() << " microseconds\n";

			hidden_layers[i] = add(matrixproduct, biases[i]);
			outputs[i] = getRelu(getLayer(i));

		}
		std::cout << "gpu_mm - time taken: ";
		auto start = std::chrono::high_resolution_clock::now();

		mat<double> matrixproduct = gpu_mm(weights[i], getLayer(i-1));
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << duration.count() << " microseconds\n";

		hidden_layers[i] = add(matrixproduct, biases[i]);
		outputs[i] = getSoftmax(hidden_layers[i]);
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
//		std::cout << "Target output:\n" << target_output.toString(5) << std::endl;
//		std::cout << "Model output:\n" << outputs[size - 1].toString(5) << std::endl;
//		std::cout << "Error between target output and model output:\n" << delta.toString(5) << std::endl;
//		std::cin.get();
		double scalar = (double)1/(double)input_normalized.columns();
		for(int i = size-1;i>=0;i--)
		{
			if(i < size-1)
			{
				std::cout << "gpu_mm - time taken: ";
				auto start = std::chrono::high_resolution_clock::now();

				mat<double> matrixproduct = gpu_mm(getTranspose(weights[i+1]), delta);
				auto stop = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
				std::cout << duration.count() << " microseconds\n";

				delta.newValues(hadamard(matrixproduct, drelu(getLayer(i))));
			}
			mat<double> scalarproduct = scalar_m(delta, scalar);
//			std::cout << "Scalar product in bprop: \n" << scalarproduct.toString(9) << std::endl;
			std::cout << "gpu_mm - time taken: ";
			auto start = std::chrono::high_resolution_clock::now();

			weights_delta[i] = gpu_mm(scalarproduct, getTranspose(getOutput(i-1)));
			auto stop = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
			std::cout << duration.count() << " microseconds\n";

			biases_delta[i] = sum_m(scalarproduct);
		}
		for(int i = 0;i<weights_delta.size();i++)
		{
			//TESTOUTPUTS: for monitoring the deltas
			mat<double> weights_delta_sum = sum_m(getTranspose(sum_m(weights_delta[i])));
			std::cout << "Total sum of weights delta  " << i << ": " << weights_delta_sum[0][0] << "\n";
			mat<double> biases_delta_sum = sum_m(getTranspose(sum_m(biases_delta[i])));
			std::cout << "Total sum of biases delta  " << i << ": " << biases_delta_sum[0][0] << "\n";

		}
//		std::cin.get();	
			
		updateParameters(weights_delta, biases_delta);
	}

	void nnet::updateParameters(std::vector<mat<double>> weights_delta, std::vector<mat<double>> biases_delta)
	{
		for(int i = 0;i<size;i++)
		{
//			std::cout << "Weights of layer " << i << ": " << weights[i].toString(1) << std::endl;
			mat<double> scalarproduct = scalar_m(weights_delta[i], learningrate);
//			std::cout << "Scalar product of the weights " << i << " in updateParameters: " << scalarproduct.toString(2) << std::endl;
//			std::cout << "Error of layer " << i << ": " <<getError(weights[i], scalar_m(weights_delta[i], learningrate)).toString(1) << std::endl;
//			std::cout << "Error of layer " << i << " multiplied by scalar: " << scalar_m(weights_delta[i], learningrate).toString(1) << std::endl;
			weights[i] = getError(weights[i], scalarproduct);
//			std::cout << "Updated weights of layer " << i << ": " << weights[i].toString(1) << std::endl;
//			std::cout << weights[i].toString(2);
			biases[i] = getError(biases[i], scalar_m(biases_delta[i], learningrate));
//			std::cin.get();
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
		input.newValues(newinput_mat);
		input_normalized.newValues(getNormalized(newinput_mat, features_maxvalue));
	}

	mat<double>& nnet::getLayer(int i)
	{
		if(i < 0)
			return input_normalized;
		if(i < size)
			return hidden_layers[i];
		else
			return input_normalized;
	}

	const mat<double>& nnet::getLayer(int i) const
	{
		if(i < 0)
			return input_normalized;
		if(i < size)
			return hidden_layers[i];
		else
			return input_normalized;
	}

	mat<double>& nnet::getOutput(int i)
	{
		if(i < 0)
			return input_normalized;
		if(i < size)
			return outputs[i];
		else
			return input_normalized;
	}

	const mat<double>& nnet::getOutput(int i) const
	{
		if(i < 0)
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
		
	void nnet::printLayer(int i, std::ostream &o, int precision, int imagewidth)
	{
		o << getOutput(i).toString(precision, imagewidth);
	}

	void nnet::resetParameters()
	{
		//Set the weights to new values using the He-method
		for (auto& w : weights)
		{
			w.heInitialize((double)w.columns(), 0.0);
		}
		//Set the biases to 0.1
		for (auto& b : biases)
		{
			for (auto& i : b.m)
			{
				for (auto& j : i)
				{
					j = 0.1;
				}
			}
		}
	}
}
