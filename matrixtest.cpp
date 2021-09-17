#include "feedforward_network.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include "onehot.h"
#include <memory>
#include <fstream>
#include "dataloader.h"

//Constants

#define PRECISION(VAL) std::fixed << std::setprecision(VAL)
#define TEST std::cout << "test!" << std::endl

int main(int argc, char *argv[])
{
	//Declare the network with default constructor
	MM::nnet network;
	//Variable showing the state of the network
	bool network_loaded = false;

	//User inputs
	int user_input_c = argc - 1;
	std::string user_input_line;
	std::string user_input_word;
	std::vector<std::string> user_input(user_input_c);
	for(int i = 1;i<user_input_c;i++)
	{
		user_input[i-1] = std::string(argv[i]);
	}
	std::string modelName;
	int epoch = -1;
	int batchsize = -1;
	int features_maxvalue = -1;
	int layersize = -1;
	double learningrate = -1;
	int imagewidth = -1;
	bool quit = false;
	//If the user wants to output to a file, this turns true
	bool output_to_file = false;
	bool showLabels = false;
	std::ofstream output;

	std::vector<int> images;
	std::vector<int> labels;
	bool images_loaded = false;
	bool labels_loaded = false;

	//Process the possible command line arguments
	std::vector<std::string> userInput = readCommand(argc, argv);
	while(!quit)
	{
		readCommand(std::cin);
		switch(userInput[0])
		{
			case QUIT:
				quit = true;
				break;
			case HELP:
				printHelp(userInput
				break;
			case PREDICT:
				break;
			case LOAD_INPUT:
				break;
			case LOAD_LABELS:
				break;
			case TRAIN_MODEL:
				break;
			case LOAD_MODEL:
				break;
			case SETTINGS:
				break;
			case HELP:
				printHelp(userInput[1]);
				break;
			default:
				break;
	}
	return 0;
}
