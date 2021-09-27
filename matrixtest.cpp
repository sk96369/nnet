#include "feedforward_network.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include "onehot.h"
#include <memory>
#include <fstream>
#include "dataloader.h"

//Constants
#define VERSION 1
//Size of the vector that holds the metadata of input data
#define METADATA_SIZE 3
#define PRECISION(VAL) std::fixed << std::setprecision(VAL)
#define TEST std::cout << "test!" << std::endl

int main(int argc, char *argv[])
{
	//Declare the network with default constructor
	MM::nnet network;
	//Declare a ProgramSettings object
	MM::ProgramSettings settings;
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
	//Vector that holds the metadata
	std::vector<int> label_metadata(METADATA_SIZE);
	std::vector<int> images_metadata(METADATA_SIZE);
	bool images_loaded = false;
	bool labels_loaded = false;

	//Load the last used settings
	loadSettings(0);
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
			case EVALUATE:
				if(network_loaded)
				{
					if(images_loaded)
					{
						std::vector<int> predictions = network.predict(images);
						std::cout << "Select the file to evaluate the model outputs with:\n";
						std::string filename;
						std::vector<std::string> filelist = MM::listFiles("labels", MM::listInputFormats(VERSION));

						if()
						{
							std::cout << 
							if(!loadData(user_input[1], labels_metadata, labels))
							{
								std::cout << "Could not load label data!\n";
							}
							if(labels_metadata[ITEMS] == images_metadata[ITEMS])
							{
								int correct = 0;
								for(int i = 0;i<labels.size();i++)
								{
									if(predictions[i] == labels[i])
										correct++;
									std::cout << "Prediction " << i << ": " << predictions[i] << " - Ground truth: " << labels[i] << std::endl;
								}
								std::cout << "Accuracy: " << ((double) correct/(double)labels.size()) * 100 << "%\n";
							}
							else
							{
								std::cout << "The number of input data points and labels do not match.\n";
							}
						}
						else
						{
							for(int i = 0;i<network.getDimension(-1);i++)
							{
								std::cout << "Prediction " << i << ": " << predictions[i] << std::endl;
							}
						}
						std::cout << std::endl;
					}
					else
						std::cout << "No predictions can be made because no inputs have been loaded yet. To load inputs, use command \"loadinput\".\n";
				}
				else
				{
					std::cout << "No predictions can be made because the network has not been initialized yet. To initialize the network, use command \"trainmodel\" or \"loadmodel\".\n";
				}
				break;

			case PREDICT:
				if(network_loaded)
				{
					if(images_loaded)
					{
						std::vector<int> predictions = network.predict(images);
						if(parseCommand)
						{
							if(!loadData(user_input[1], labels_metadata, labels))
							{
								std::cout << "Could not load label data!\n";
							}
							if(labels_metadata[ITEMS] == images_metadata[ITEMS])
							{
								int correct = 0;
								for(int i = 0;i<labels.size();i++)
								{
									if(predictions[i] == labels[i])
										correct++;
									std::cout << "Prediction " << i << ": " << predictions[i] << " - Ground truth: " << labels[i] << std::endl;
								}
								std::cout << "Accuracy: " << ((double) correct/(double)labels.size()) * 100 << "%\n";
							}
							else
							{
								std::cout << "The number of input data points and labels do not match.\n";
							}
						}
						else
						{
							for(int i = 0;i<network.getDimension(-1);i++)
							{
								std::cout << "Prediction " << i << ": " << predictions[i] << std::endl;
							}
						}
						std::cout << std::endl;
					}
					else
						std::cout << "No predictions can be made because no inputs have been loaded yet. To load inputs, use command \"loadinput\".\n";
				}
				else
				{
					std::cout << "No predictions can be made because the network has not been initialized yet. To initialize the network, use command \"trainmodel\" or \"loadmodel\".\n";
				}
				break;
			case LOAD_INPUT:
				std::cout << "Choose a file to load the input from:\n";
				std::string inputFileName;
				if(selectFile("input", inputFileName))
				{
					
						if(!loadData(filenames[selection], images_metadata, images))
						{
							std::cout << "Input data could not be loaded!\n";
							//                                         --------------Print additional information
						}
						else
						{
							imagewidth = images_metadata[COLUMNS];
							images_loaded = true;
						}
					}
				}
				else
				{
					std::cout << "No valid input data found! Place the input files in the Data folder\nValid input data formats for version " << VERSION << ":\n";
					for(auto& format : formats)
					{
						std::cout << "   " << format << std::endl;
					}
				}
				break;
			case LOAD_LABELS:
				std::cout << "Choose a file to load the labels from:\n";
				std::vector<std::string> formats = listInputFormats(VERSION);
				std::vector<std::string> filenames = listFiles(formats);
				if(filenames.size() > 0)
				{
					int selection = -1;
					while(selection < 0)
					{
						for(int i = 0;i < filenames.size();i++)
						{
							std::cout << i << ": " << filename << std::endl;
						}
						if(!readInt(std::cin, selection))
						{
							std::cout << "Invalid input!\n";
						}
					}
					if(!loadData(filenames[selection], labels_metadata, labels))
					{
						std::cout << "Label data is invalid!\n";
					}
					else
						labels_loaded = true;
				}
				else
				{
					std::cout << "No valid input data found! Place the input files in the Data folder\nValid input data formats for version " << VERSION << ":\n";
					for(auto& format : formats)
					{
						std::cout << "   " << format << std::endl;
					}
				}
				break;
			case TRAIN_MODEL:
				if(images_loaded && labels_loaded)
				{
					if(images_metadata[ITEMS] == labels_metadata[ITEMS])
					{
						if(user_input_c > 6)
						{
							std::stringstream ss;
							modelName = user_input[1];
							ss << user_input[2] << " " << user_input[3]<< " " << user_input[4] << " " << user_input[5];
							if(ss >> batchsize)
								std::cout << "Batch size: " << batchsize << "\n";
							else
							{
								std::cout << "No integer found for batch size, exiting...\n";
								exit(1);
							}
							if(ss >> epoch)
								std::cout << "Training epochs: " << epoch << "\n";
							else
							{
								std::cout << "No integer found for training epoch count, exiting...\n";
								exit(1);
							}
							if(ss >> features_maxvalue)
								std::cout << "Max value of pixel in training data: " << features_maxvalue << "\n";
							else
							{
								std::cout << "No integer found for batch size, exiting...\n";
								exit(1);
							}
							if(ss >> learningrate)
								std::cout << "Learning rate: " << learningrate << "\n";
							else
							{
								std::cout << "No floating point value found for learning rate, exiting...\n";
								exit(1);
							}
				
							//Read the sizes of the hidden layers
							std::vector<int> dimensions(user_input_c - 6);
							std::stringstream dimension_ss;
							for(int i = 6;i<user_input_c;i++)
							{
								dimension_ss << user_input[i] << " ";
							}
							for(int i = 6;i<user_input_c;i++)
							{
								int dimension = -1;
								dimension_ss >> dimension;
								if(dimension > 0)
								{
									dimensions[i-6] = dimension;
								}
								else
								{
									std::cout << "Could not read size of hidden layer " << i-6 << ". Exiting...\n";
									exit(1);
								}
							}
				
							int inputsize = images.size()/labels.size();
							std::vector<int> dimensions_with_input;
							dimensions_with_input.push_back(inputsize);
							for(auto& i : dimensions)
							{
								dimensions_with_input.push_back(i);
							}

							//If a network has not yet been constructed, call the constructor with the given dimensions
							if(!network_loaded)
								network = MM::nnet(dimensions_with_input, features_maxvalue, learningrate);
							if(batchsize > labels.size() || batchsize < 1)
								batchsize = labels.size();
							network.trainRandom(images, labels, batchsize, inputsize, labels.size(), epoch, showLabels, imagewidth);
							network.saveModel(user_input[1]);
							network_loaded = true;
						}
					}
					else
					{
						std::cout << "The input data and training labels do not match!\n";
					}
				}
				else
					std::cout << "Training the model requires images and labels to be loaded. Type \"help\" to get help.\n";
			}

				break;
			case LOAD_MODEL:
				std::vector<std::string> modelFiles = listFiles("models", ".mm");
				int modelIndex = makeSelection(modelFiles, 3);
				network.loadModel(modelFiles[modelIndex]);
				break;
			case SETTINGS:
				break;
			case HELP:
				printHelp(userInput[1]);
				break;
			case RESET_PARAMETERS:
				if(MM::confirm)
				{
					network.resetParameters();
				}
				break;
			default:
				break;
	}
	return 0;
}
