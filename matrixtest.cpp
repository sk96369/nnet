#include "gzreader.h"
#include "feedforward_network.h"
#include <iostream>
#include <vector>
#include <iomanip>

//Constants  ---PLACEHOLDER FOR USER INPUT---
const int EPOCH = 1;
const int DATASIZE = 60000;
const int BATCHSIZE = 500;
const int IMAGESIZE = 784;
const int IMAGEWIDTH = 28;
const int FEATURES_MAXVALUE = 255;

#define PRECISION(VAL) std::fixed << std::setprecision(VAL)
#define TEST std::cout << "test!" << std::endl

int main(int argc, char *argv[])
{
TEST;
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

	MM::mat<double> weight1(-1, 1, 10, 5);
	MM::mat<double> input(0, 7, 10);
	MM::mat<double> h1(0, 7, 5);
	MM::mat<double> bias1(-0.1, 0.1, 1, 5);
	MM::mat<double> weight2(-1, 1, 5, 3);
	MM::mat<double> h2(0, 7, 3);
	MM::mat<double> bias2(-0.1, 0.1, 1, 3);
	MM::mat<double> output(0, 7, 3);

	std::vector<double> input_vector;
	for(int i = 1;i<71;i++)
	{
		input_vector.push_back(1/(double)i);
	}

	input.newValues(input_vector);
	std::cout << "input:\n" << input.toString() << std::endl;
	std::cout << "weight1:\n"<< weight1.toString() << std::endl;
	std::cout << "h1:\n"<< h1.toString() << std::endl;
	std::cout << "weight2:\n"<< weight2.toString() << std::endl;
	std::cout << "h2:\n"<< h2.toString() << std::endl;

	MM::mat<double> h1_biased;
	MM::mat<double> relu;
	MM::mat<double> h2_biased;

	for(int i = 0;i<100;i++)
	{
	h1.newValues(MM::mm(weight1, input));
	h1_biased = MM::add(h1, bias1);
	relu = MM::getRelu(h1_biased);
	h2.newValues(MM::mm(weight2, relu));
	h2_biased = (MM::add(h2, bias2));
	output.newValues(MM::getSoftmax(h2_biased));

	std::cout << "input:\n" << input.toString(2) << std::endl;
	std::cout << "weight1:\n" << weight1.toString(2) << std::endl;
	std::cout << "h1:\n" << h1.toString(2) << std::endl;
	std::cout << "h1 after adding bias:\n" << h1_biased.toString(2) << std::endl;
	std::cout << "h1 after relu:\n" << relu.toString(2) << std::endl;
	std::cout << "weight2:\n" <<weight2.toString(2) << std::endl;
	std::cout << "h2:\n" << h2.toString(2) << std::endl;
	std::cout << "h2 after adding bias:\n" << h2_biased.toString(2) << std::endl;
	std::cout << "output after softmax\n" << output.toString(2) << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";

	MM::mat<double> target(0, 7, 3);
	for(int i = 0;i<7;i++)
	{
		target[i][1] = 1;
	}
	MM::mat<double> delta2(MM::getError(target, output));
	MM::mat<double> deltaw2(MM::mm(MM::scalar_m(delta2, 1/(double)7), MM::getTranspose(relu)));
	MM::mat<double> deltab2(MM::scalar_m(MM::sum_m(delta2), 1/(double)7));
	MM::mat<double> delta1(MM::hadamard(MM::mm(MM::getTranspose(weight2), delta2), MM::drelu(h1_biased)));
	MM::mat<double> deltaw1(MM::mm(MM::scalar_m(delta2, 1/(double)7), getTranspose(input)));
	MM::mat<double> deltab1(MM::scalar_m(MM::sum_m(delta1), 1/(double)7));

	std::cout << "delta2*(1/7):\n" << MM::scalar_m(delta2, 1/(double)7).toStringFlipped(4) << std::endl;
	std::cout << "delta2:\n" << delta2.toStringFlipped(2) << std::endl;
	std::cout << "relu transposed:\n" << MM::getTranspose(relu).toStringFlipped(2);

	std::cout << "deltaw2:\n" << deltaw2.toString(4) << std::endl;
	std::cout << "deltab2:\n" << deltab2.toString(4) << std::endl;
	std::cout << "delta1:\n" << delta1.toString(4) << std::endl;
	std::cout << "deltaw1:\n" << deltaw1.toString(4) << std::endl;
	std::cout << "deltab1:\n" << deltab1.toString(4) << std::endl;

	weight1.newValues(MM::getError(weight1, MM::scalar_m(deltaw1, 0.1)));
	bias1.newValues(MM::getError(bias1, MM::scalar_m(deltab1, 0.1)));
	weight2.newValues(MM::getError(weight2, MM::scalar_m(deltaw2, 0.1)));
	bias2.newValues(MM::getError(bias2, MM::scalar_m(deltab2, 0.1)));

	std::cout << "input:\n" << input.toString() << std::endl;
	std::cout << "weight1:\n"<< weight1.toString() << std::endl;
	std::cout << "h1:\n"<< h1.toString() << std::endl;
	std::cout << "weight2:\n"<< weight2.toString() << std::endl;
	std::cout << "h2:\n"<< h2.toString() << std::endl;
	std::cin.get();
}

	/* train test */
//	network.train(images, labels, BATCHSIZE, IMAGESIZE, IMAGEWIDTH, DATASIZE, EPOCH);

//	std::cout << "Final network output:\n" << network.getOutput().toString() << std::endl;
//	network.saveModel(argv[1]);
	return 0;
}
