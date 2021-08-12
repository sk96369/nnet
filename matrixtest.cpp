#include "gzreader.h"
#include "feedforward_network.h"
#include <iostream>
#include <vector>

//Constants  ---PLACEHOLDER FOR USER INPUT---
const int EPOCH = 1;
const int DATASIZE = 60000;
const int BATCHSIZE = 500;
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

	MM::mat<int> a(1, 5, 10);
	MM::mat<int> b(1, 10, 20);
	MM::mat<int> c(1, 10, 5);
	MM::mat<double> d(1.0, 20, 10);
	MM::mat<double> e(1.0, 10, 20);
	MM::mat<double> f(5.0, 20, 10);
	MM::mat<double> g;
	std::cout << "test1\n";
	g.newValues(mm(a, c));
	std::cout << "test2\n";
	printf("AC: %i %i\n", g.columns(), g.rows());
	g.newValues(mm(b, a));
	printf("BA: %i %i\n", g.columns(), g.rows());
	g.newValues(mm(d, e));
	printf("DE: %i %i\n", g.columns(), g.rows());
	g.newValues(getError(f, d));
	printf("F-D: %i %i\n", g.columns(), g.rows());

	printf("%s\n\n\n", g.toString().c_str());

	/* train test */
	network.train(images, labels, BATCHSIZE, IMAGESIZE, IMAGEWIDTH, DATASIZE, EPOCH);

	network.saveModel(argv[1]);
	return 0;
}
