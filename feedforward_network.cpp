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
	/* MEMBERS OF THE CLASS FOR REFERENCE */
//	mat<int> input;
//	std::vector<mat<double>> hidden_layers;
//	std::vector<mat<double>> biases;
//	mat<double>output;

	nnet::nnet(std::vector<int> dimensions) : learningrate(0.1), size(dimensions.size()), hidden_layers(dimensions.size() - 2), weights(dimensions.size()), input(dimensions[0]), output(dimensions[dimensions.size() - 1])
	{
		for(int i = 1;i<size - 1;i++)
		{
			mat<double> newlayer(dimensions[i]);
			hidden_layers[i-1] = newlayer;
		}

		for(int i = 1;i<size;i++)
		{
			mat<double> newweight(-0.5, 0.5, *this[i-1].size(), *this[i].size());
			weights[i-1] = newweight;
		}
		for(int i = 1;i<size;i++)
		{
			mat<double> newbias(-0.2, 0.2, 0, *this[i].size());
			biases[i-1] = newbias;
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

			input.newValues(imagebatch);

			//Move the iterators
			image_start += batchsize*imagesize;
			label_start += batchsize;
			if(image_start != images.end())
			{
				image_end += batchsize*imagesize;
				label_end += batchsize;
			}
			
	}
	void nnet::fprop()
	{

	}
	void nnet::bprop()
	{
	}

	std::vector<int> nnet::getDimensions()
	{
		std::vector<int> dimensions(size);
		dimensions[0] = input.size();
		for(int i = 1;i<size - 1;i++)
		{
			dimensions[i] = hidden_layers[i-1].size();
		}
		dimensions[size-1] = output.size();
		return dimensions;
	}

	mat<A>& operator[](int i)
	{
		if(i == 0)
		{
			return input;
		}
		else if(i == size-1)
		{
			return output;
		}
		return hidden_layers[i];
	}

	const mat<A>& operator[](int i) const
	{
		if(i == 0)
		{
			return input;
		}
		else if(i == size-1)
		{
			return output;
		}
		return hidden_layers[i];
	}

}
