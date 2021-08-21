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
	std::ofstream output;

	std::vector<int> images;
	std::vector<int> labels;
	bool images_loaded = false;
	bool labels_loaded = false;

	while(!quit)
	{
		if(user_input_c > 0)
		{
			if(user_input[0] == "imagewidth" && user_input_c > 1)
			{
				std::stringstream ss;
				ss << user_input[1];
				ss >> imagewidth;
			}

			if((user_input[0] == "print" || user_input[0] == "p"))
			{
				if(network_loaded)
				{
					std::stringstream ss;
					int layer = 0;
					int precision = 0;
					if(user_input_c > 1)
					{
						if(user_input[1] == "input")
						{
							if(imagewidth > 0)
							{
								std::string inputmatrix;
								int in_row = 0;
								for(auto& ptr : images)
								{
									if(ptr == 0)
										inputmatrix.append(" ");
									else
										inputmatrix.append("#");
									in_row++;
									if(in_row == imagewidth)
									{
										in_row = 0;
										inputmatrix.append("\n");
									}
								}
								if(output_to_file)
								{
									int i = 0;
									for(auto& symbol : inputmatrix)
									{
										output << symbol;
										i++;
										if(i == imagewidth)
										{
											output << "\n";
											i = 0;
										}
									}
								}
								else
								{
									int i = 0;
									for(auto& symbol : inputmatrix)
									{
										std::cout << symbol;
										i++;
										if(i == imagewidth)
										{
											std::cout << "\n";
											i = 0;
										}
									}
								}
							}
							else
								std::cout << "You need to set image width first.\n";
						}
						else if(user_input[1] == "output")
						{
							std::vector<int> outputs = onehot_toInt(network.getOutput());
							if(output_to_file)
							{
								for(int i = 0;i<outputs.size();i++)
								{
									output << "Output for data point " << i << ": " << outputs[i] << "\n";
								}
							}
							else
							{
								for(int i = 0;i<outputs.size();i++)
								{
									std::cout << "Output for data point " << i << ": " << outputs[i] << "\n";
								}
							}
						}
						else
						{
							if(ss << user_input[1])
							{
								precision = 2;
								ss >> layer;
								if(user_input_c > 2)
								{
									ss >> precision;
									if(precision < 0)
										precision = 0;
								}
								if(output_to_file)
									network.printLayer(layer, output, precision);
								else
									network.printLayer(layer, std::cout, precision);
							}
							else
							{
								std::cout << "Print format: print [layer] ([precision])\n";
							}
						}
					}
				}
				else
					std::cout << "The network needs to be initialized before printing. Type \"help\" for help.\n";
			}
							
			if(user_input[0] == "tm" || user_input[0] == "trainmodel")
			{
				if(images_loaded && labels_loaded)
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
						network.trainRandom(images, labels, batchsize, inputsize, labels.size(), epoch);
						network.saveModel(user_input[1]);
						network_loaded = true;
					}
				}
				else
					std::cout << "Training the model requires images and labels to be loaded. Type \"help\" to get help.\n";
			}

			if(user_input[0] == "loadmodel" || user_input[0] == "lm")
			{
				if(user_input_c > 1)
				{
					network = MM::nnet(std::string(user_input[1]));
					network_loaded = true;
				}
				else
					std::cout << "loadmodel requires the following arguments:\n   loadmodel [filename]\n";
			}
				
			if(user_input[0] == "loadinput" || user_input[0] == "li")
			{
				if(user_input_c > 1)
				{
					images = loadData(user_input[1]);
					if(images.size() > 0)
						images_loaded = true;
				}
				else
					std::cout << "loadinput requires the following arguments:\n   loadinput [filename]\n";
			}

			if(user_input[0] == "loadlabels" || user_input[0] == "ll")
			{
				if(user_input_c > 1)
				{
					labels = loadData(user_input[1]);
					if(labels.size() > 0)
						labels_loaded = true;
				}
				else
					std::cout << "loadlabels requires the following arguments:\n   loadlabels [filename]\n";
			}
	
			if(user_input[0] == "predict")
			{
				if(network_loaded)
				{
					if(images_loaded)
					{
						std::vector<int> predictions = network.predict(images);
						if(user_input_c > 1)
						{
							labels = loadData(user_input[1]);
							for(int i = 0;i<labels.size();i++)
							{
								std::cout << "Prediction " << i << ": " << predictions[i] << " - Ground truth: " << labels[i] << std::endl;
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
			}
		}

		//Read user input
		std::cout << "Type a command: (\"help\" to list available commands)\n";
		std::getline(std::cin, user_input_line);
		std::stringstream user_input_ss;
		user_input_ss << user_input_line;
		user_input.clear();
		user_input_c = 0;
		while(user_input_ss >> user_input_word)
		{
			user_input_c ++;
			user_input.push_back(user_input_word);
		}
		if(user_input[0] == "quit" || user_input[0] == "q")
		{
			quit = true;
		}
		if(user_input[0] == "output")
		{
			if(user_input_c > 1)
			{
				if(user_input[1] == "default")
					output_to_file = false;
				else
				{
					output_to_file = true;
					if(output.is_open())
						output.close();
					output.open(user_input[1]);
				}
			}
			else
			{
				output_to_file = false;
				if(output.is_open())
					output.close();
			}
		}
		if(user_input[0] == "help")
		{
			if(user_input_c == 1)
			{
				std::cout << "_________________\nimagewidth [width of input image]\nloadinput, li [filename]\nloadlabels, ll [filename]\nloadmodel, lm [filename]\npredict ([filename])\nprint [layer] ([precision])\nquit, q\noutput [filename, \"default\"]\ntrainmodel [model name] [batch size] [training epochs] [maximum pixel value] [learning rate] [size of layer z1] ... [size of layer zn (output layer)]\n________________\n\"help [command] to show more information on a command.\n";
			}
			else
			{
				if(user_input[1] == "imagewidth")
				{
					std::cout << "Sets the width of the input images for printing purposes.\nRequired arguments: [width of input image]\nThe argument [width of input image] takes in an integer value over 0.\n";
				}
				if(user_input[1] == "loadinput")
				{
					std::cout << "Reads input data from a file with the given filename for the model to use. This command is a prerequisite requirement for making predictions and training the model.\nRequired arguments: [filename]\n";
				}
				if(user_input[1] == "loadlabels")
				{
					std::cout << "Reads ground truth data from the given file to use in adjusting parameters. This command is a prerequisite requirement for training the model.\nRequired arguments: [filename]\n";
				}
				if(user_input[1] == "loadmodel")
				{
					std::cout << "Reads model dimensions and parameters from the given file. Initializes the model.\nRequired arguments: [filename]\n";
				}
				if(user_input[1] == "predict")
				{
					std::cout << "Calculates outputs based on the loaded input data.\nThe optional [filename] argument loads ground truth data from a file with the given name to print alongside the model's predictions.\n   Requires a model to be initialized.\n";
				}
				if(user_input[1] == "print")
				{
						std::cout << "Prints the matrix at the given layer of the model.\nRequired arguments: [layer] ([precision])\nThe [layer] argument takes in a number between -1 and n, with -1 being the index of the input matrix, and n being the index of the final hidden layer, aka. the output layer.\nThe [layer] argument also accepts keywords \"input\" and \"output\", with input printing out the inputs as images, and output printing out the model's predictions as integers.\nThe optional [precision] argument takes in an integer signifying how many numbers after the decimal point are printed.\n   Requires a model to be initialized.\n";
				}
				if(user_input[1] == "quit")
				{
					std::cout << "Exits the program\n";
				}
				if(user_input[1] == "output")
				{
					std::cout << "Sets the program's output stream to a file with the given filename, or sets it to default, to print output to the console.\n";
				}
				if(user_input[1] == "trainmodel")
				{
					std::cout << "Adjusts the weights of the model based on the loaded input and label data. Runtime might be long.\nRequired arguments: [model name] [batch size] [training epochs] [maximum pixel value] [learning rate] [size of layer z1] ... [size of layer zn (output layer)]\nThe [model name] argument takes in the desired name of the text file in which to save the trained model parameters.\nThe [batch size] argument takes in an integer signifying how many training examples are propagated at a time.\nThe [training epochs] argument takes in an integer signifying how many times the entire training data is passed through the network.\nThe [maximum pixel value] tells the model what the range of values for each pixel is.\nThe [learning rate] affects the amount each parameter is changed during training. This value should be a positive real number greater than 0 and less than 1.\nThe [size of layerz1] ... [size of layer zn] arguments signify how many units are in each hidden layer of the model. The minimum number of hidden layers is 1.\nInitializes the model\n   Requires input and labels to be loaded.\n";
				}
				else
					std::cout << user_input[1] << " is not a recognized command.\n";
			}
		}
	}
	return 0;
}
