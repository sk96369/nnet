const int EPOCH = 1;
const int DATASIZE = 60000;
const int BATCHSIZE = 1000;
const int IMAGESIZE = 784;
const int IMAGEWIDTH = 28;

#include "feedforward_network.h"
#include <iostream>
#include <vector>

int main()
{

	std::vector<int> input;
	std::vector<int> dimensions = {IMAGESIZE, 10, 10}; //READ FROM INPUT

	std::vector<int> images = readmnistgz("imagedata.gz");
	std::vector<int> labels = readmnistgz("traininglabels.gz");

	MM::nnet network(dimensions);
	std::vector<int> received_dimensions = network.getDimensions();
	for(auto& i : received_dimensions)
	{
		std::cout << i << " ";
	}
	std::cout << "\n";

	/* train test */
	network.train(images, labels, BATCHSIZE, IMAGESIZE, IMAGEWIDTH, DATASIZE, EPOCH);

	return 0;
}
