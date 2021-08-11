#include "gzreader.h"
#include "feedforward_network.h"
#include <iostream>
#include <vector>

//Constants  ---PLACEHOLDER FOR USER INPUT---
const int EPOCH = 1;
const int DATASIZE = 60000;
const int BATCHSIZE = 100;
const int IMAGESIZE = 784;
const int IMAGEWIDTH = 28;
const int FEATURES_MAXVALUE = 255;


int main(int argc, char *argv[])
{

	std::vector<int> input;
	std::vector<int> dimensions = {IMAGESIZE, 10, 10}; //READ FROM INPUT

	std::vector<int> images = readmnistgz("imagedata");
	std::vector<int> labels = readmnistgz("traininglabels");

	MM::nnet network(dimensions, FEATURES_MAXVALUE);
	std::vector<int> received_dimensions = network.getDimensions();
	std::cout << "Network's dimensions: \n";
	for(auto& i : received_dimensions)
	{
		std::cout << i << " ";
	}
	std::cout << "\n";

	/* train test */
	network.train(images, labels, BATCHSIZE, IMAGESIZE, IMAGEWIDTH, DATASIZE, EPOCH);

	network.saveModel(argv[1]);
	return 0;
}
