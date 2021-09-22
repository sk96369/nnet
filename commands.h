#include <iostream>
#include <stdexcept>
#include <string>

namespace MM
{
	enum Commands
	{
		NONE,
		HELP,
		QUIT,
		LOAD_INPUT,
		LOAD_LABELS,
		TRAIN_MODEL,
		LOAD_MODEL,
		PRINT,
		SETTINGS,
		RESET_PARAMS,
		PREDICT,
	}

	bool parseCommands(std::queue<int> &commandQueue, int &output)
	{
		if(!commandQueue.empty())
		{
			output = commandqueue.front();
			commandqueue.pop();
			return true;
		}
		return false;
	}

	void printHelp(int command)
	{
		switch(command)
		{
			case NONE:
				std::cout << "imagewidth [width of input image]\nloadinput, li [filename]\nloadlabels, ll [filename]\nloadmodel, lm [filename]\npredict ([filename])\nprint [layer] ([precision])\nquit, q\noutput [filename, \"default\"]\ntogglelabels\ntrainmodel [model name] [batch size] [training epochs] [maximum pixel value] [learning rate] [size of layer z1] ... [size of layer zn (output layer)]\n________________\n\"help [command] to show more information on a command.\n";
				break;
			case SETTINGS:


			else
			{
				if(user_input[1] == "imagewidth")
				{
					std::cout << "imagewidth [width of input image]\nSets the width of the input images for printing purposes.\nThe argument [width of input image] takes in an integer value over 0.\n";
				}
				if(user_input[1] == "loadinput")
				{
					std::cout << "Reads input data from a file with the given filename for the model to use. This command is a prerequisite requirement for making predictions and training the model.\nRequired arguments: [filename]\n";
				}
				if(user_input[1] == "loadlabels")
				{
					std::cout << "loadlabels [filename]\nReads ground truth data from the given file to use in adjusting parameters. This command is a prerequisite requirement for training the model.\n";
				}
				if(user_input[1] == "loadmodel")
				{
					std::cout << "loadmodel [filename]\nReads model dimensions and parameters from the given file. Initializes the model.\n";
				}
				if(user_input[1] == "predict")
				{
					std::cout << "predict ([filename])\nCalculates outputs based on the loaded input data.\nThe optional [filename] argument loads ground truth data from a file with the given name to print alongside the model's predictions.\n   Requires a model to be initialized.\n";
				}
				if(user_input[1] == "print")
				{
						std::cout << "print [layer] ([precision])\nPrints the matrix at the given layer of the model.\nThe [layer] argument takes in a number between -1 and n, with -1 being the index of the input matrix, and n being the index of the final hidden layer, aka. the output layer.\nThe [layer] argument also accepts keywords \"input\" and \"output\", with input printing out the inputs as images, and output printing out the model's predictions as integers.\nThe optional [precision] argument takes in an integer signifying how many numbers after the decimal point are printed.\n   Requires a model to be initialized.\n";
				}
				if(user_input[1] == "quit")
				{
					std::cout << "Exits the program\n";
				}
				if(user_input[1] == "output")
				{
					std::cout << "Sets the program's output stream to a file with the given filename, or sets it to default, to print output to the console.\n";
				}
				if(user_input[1] == "togglelabels")
				{
						std::cout << "Toggles the showLabels setting. When training a model when switched on, the training images and labels will be printed.\n   Requires imagewidth to be set.\n";
				}
				if(user_input[1] == "trainmodel")
				{
					std::cout << "trainmodel [model name] [batch size] [training epochs] [maximum pixel value] [learning rate] [size of layer z1] ... [size of layer zn (output layer)]\nAdjusts the weights of the model based on the loaded input and label data. Runtime might be long.\nThe [model name] argument takes in the desired name of the text file in which to save the trained model parameters.\nThe [batch size] argument takes in an integer signifying how many training examples are propagated at a time.\nThe [training epochs] argument takes in an integer signifying how many times the entire training data is passed through the network.\nThe [maximum pixel value] tells the model what the range of values for each pixel is.\nThe [learning rate] affects the amount each parameter is changed during training. This value should be a positive real number greater than 0 and less than 1.\nThe [size of layerz1] ... [size of layer zn] arguments signify how many units are in each hidden layer of the model. The minimum number of hidden layers is 1.\nInitializes the model\n   Requires input and labels to be loaded.\n";
				}
				else
					std::cout << user_input[1] << " is not a recognized command.\n";
			}
		}
	}

	bool readInt(std::istream &is, int &output, int min = 0, int max = 0)
	{
		std::string inputString;
		getline(is, inputString);
		try
		{
			output = std::stoi(inputString);
			if(min < max && (output < min || output > max))
				throw std::out_of_range;
		}
		catch(std::invalid_argument)
		{
			std::cout << "ERROR: Not an integer!\n";
			return false;
		}
		catch(std::out_of_range)
		{
			printf("ERROR: Integer out of range (%i - %i)\n", min, max);
			return false;
		}
		return true;
	}

	bool readString(std::istream &is, std::string &output)
	{
		getline(is, inputString);
		return true;
	}

	std::queue<int> readCommand(istream &is)
	{
		std::queue<int> commands;
		//Read user input
		std::cout << "Type a command: (\"help\" to list available commands)\n>";
		std::getline(std::cin, user_input_line);
		std::stringstream user_input_ss;
		std::string word << user_input_line;
		
		if(user_input[0] == "quit" || user_input[0] == "q")
		{
			commands.push_back(QUIT);
		}
		if(user_input[0] == "settings")
		{
			commands.push_back(SETTINGS);
		}
		if(user_input[0] == "help")
		{
			commands.push_back(HELP);
		}

		if((user_input[0] == "print" || user_input[0] == "p"))
		{
			commands.push_back(PRINT);
		}
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
							if(output_to_file)
							{
								network.printLayer(-1, output, -1, imagewidth);
							}
							else
							{
								network.printLayer(-1, std::cout, -1, imagewidth);
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
				commands.push_back(TRAIN_MODEL);
				
			if (user_input[0] == "resetparameters" || user_input[0] == "rp")
			{
				commands.push_back(RESET_PARAMETERS);
			}

			if(user_input[0] == "loadmodel" || user_input[0] == "lm")
			{
				commands.push_back(LOAD_MODEL);
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
				commands.push_back(LOAD_INPUT);
			}

			if(user_input[0] == "loadlabels" || user_input[0] == "ll")
			{
				commands.push_back(LOAD_LABELS);
			}
	
			if(user_input[0] == "predict")
			{
				commands.push_back(PREDICT);
				
			}
			if(user_input[0] == "togglelabels")
			{
				if(imagewidth > 0)
				{
					showLabels += 1 - 2 * showLabels;
					std::cout << "showLabels = " << showLabels << std::endl;
				}
				else
				{
					std::cout << "You need to set imagewidth to be able to print the training images and labels\n";
				}
			}
		//Add the rest of the command to the queue
		while(user_input_ss >> user_input_word)
		{
			commands.push_back(user_input_word);
		}
	}
}
