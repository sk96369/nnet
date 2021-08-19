#include "gzreader.h"
#include "feedforward_network.h"
#include <iostream>
#include <vector>
#include <iomanip>

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
	std::stringstream user_input_ss;
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
	bool quit = false;
	std::ostream output = std::cout;

	std::vector<int> images;
	std::vector<int> labels;
	bool images_loaded = false;
	bool labels_loaded = false;

	while(!quit)
	{
		if(user_input_c > 0)
		{
			if((user_input[0] == "print" || user_input[0] == "p") && network_loaded)
			{
				std::stringstream ss;
				int layer = 0;
				int precision = 0;
				if(user_input_c > 1)
				{
					switch (user_input[1])
					{
						case "input":
							precision = 1;
							if(user_input_c > 2)
							{
								ss << user_input[2];
								ss >> precision;
							}
							network.printLayer(-1, output, precision);
							break;
						case "output":
							if(user_input_c > 2)
							{
								ss << user_input[2];
								ss >> precision;
							}
							network.print(network.getDimensions().size()-1, output, precision);
							break;
						default:
							if(ss << user_input[1])
							{
								precision = 2;
								ss >> layer;
								if(user_input_c > 2)
									ss >> precision;
								network.print(layer, output, precision);
							}
							else
							{
								std::cout << "Print format: print [layer] ([precision])\n";
							}
							break;
					}
					std::cout << "Print format: print [layer] ([precision])\n  (for more info on layers or precision, type \"help layer/precision\" \n";
				}	
	
							
			if(user_input[0] == "tnm" || user_input[0] == "trainnewmodel")
			{
				if(user_input_c > 5)
				{
					std::stringstream ss;
					modelName = user_input[1];
					ss << user_input[2] << " " << user_input[3] << " " << user_input[4];
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
		
					//Read the sizes of the hidden layers
					std::vector<int> dimensions(user_input_c-6);
					for(int i = 6;i<user_input_c;i++)
					{
						ss << user_input[i];
						if(ss >> dimension)
						{
							if(dimension < 1)
							{
								std::cout << "Dimension size must be greater than 0. Exiting...\n";
								exit(1);
							}
							dimensions[i-6] = dimension;
						else
						{
							std::cout << "Could not read size of hidden layer " << i-6 << ". Exiting...\n";
							exit(1);
						}
					}
		
					int inputsize = images.size()/labels.size();
					dimensions.push_front(inputsize);
		
					network = MM::nnet(dimensions, features_maxvalue);
					if(batchsize > labels.size() || batchsize < 1)
						batchsize = labels.size();
					network.train(images, labels, batchsize, inputsize, labels.size(), EPOCH);
					network.saveModel(user_input[1]);
				}
				else
				{
					std::cout << "trainnewmodel requires the following arguments:\n
						   trainnewmodel [model name] [batch size] [training epochs] [maximum pixel value] [size of layer z1] ... [size of layer zn (output layer)]\n";
					exit(1);
				}
			
			}

			if(user_input[0] == "loadmodel" || user_input[0] == "lm")
			{
				if(user_input_c > 1)
					network = MM::nnet(std::string(user_input[1]));
				else
					std::cout << "loadmodel requires the following arguments:\n   loadmodel [filename]\n";
			}
				
			if(user_input[0] == "loadinput" || user_input[0] == "li")
			{
				if(user_input_c > 1)
				{
					images = readmnistgz(std::string(user_input[1]), ".gz");
					if(images.size() > 0)
						images_loaded = true;
				}
				else
					std::cout << "loadinput requires the following arguments:\n   loadinput [filename]\n";
			}
	
			if(user_input[0] == "predict")
			{
				if(network_loaded)
				{
					if(images_loaded)
					{
						std::vector<int> predictions = network.predict(images);
						if(user_input_c > 4)
						{
							labels = readmnistgz(std::string(user_input[4]), ".gz");
							for(int i = 0;i<labels.size();i++)
							{
								std::cout << "Prediction: " << predictions[i] << " - Ground truth: " << labels[i] << std::endl;
							}
						}
						else
						{
							for(int i = 0;i<images.size()/;i++)
							{
								std::cout << "Prediction: " << predictions[i] << std::endl;
							}
						}
						std::cout << std::endl;
					}
					else
						std::cout << "No predictions can be made because no inputs have been loaded yet. To load inputs, use command \"loadinput\".\n";
				}
				else
				{
					std::cout << "No predictions can be made because the network has not been initialized yet. To initialize the network, use command \"trainnewmodel\" or \"loadmodel\".\n";
				}
			}
		}
		else
		{
		}
		//Read user input
		std::cout << "Type a command: (\"help\" to list available commands)\n";
		std::getline(std::cin, user_input_line);
		user_input_ss << user_input_line;
		user_input.clear();
		user_input_c = 0;
		while(user_input_ss >> user_input_word)
		{
			user_input_c++;
			user_input.push_back(user_input_word);
		}
		switch (user_input[0])
		{
			case "quit"
				quit = true;
				break;
			case "q"
				quit = true;
				break;
			case "tofile"
				if(user_input_c > 1)
				{
					output.close();
					if(user_input[1] == "default")
						output = std::cout;
					output.open(user_input[1], ios_base::app)
				}
				else
				{
					output.close();
					output = std::cout;
				}
				break;
			default
				break;
		}
	}
	return 0;
}
