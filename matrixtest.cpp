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
	//User inputs
	int user_input_c = argc - 1;
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
	bool quit = 0;

	while(!quit)
	{
		if(user_input_c > 0 && (user_input[0] == "tnm" || user_input[0] == "trainnewmodel"))
		{
			if(user_input_c > 6)
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
	
	
				std::vector<int> images = readmnistgz("imagedata");
				std::vector<int> labels = readmnistgz("traininglabels");
				int inputsize = images.size()/labels.size();
				dimensions.push_front(inputsize);
	
				MM::nnet network(dimensions, features_maxvalue);
				if(batchsize > labels.size())
					batchsize = labels.size();
				network.train(images, labels, batchsize, inputsize, labels.size(), EPOCH);
				network.saveModel(user_input[1]);
			}
			else
			{
				std::cout << "the trainnewmodel command requires the following arguments:\n
					   trainnewmodel [model name] [batch size] [training epochs] [maximum pixel value] [size of layer z1] ... [size of layer zn (output layer)]\n";
				exit(1);
			}
		
		}
		std::vector<int> eval_images;
		std::vector<int> eval_labels;
		if(strcmp(user_input[1], "predict") == 1)
		{
			if(user_input_c > 3)
			{
				eval_images = readmnistgz(std::string(user_input[2]), ".gz");
				MM::nnet network(std::string(user_input[3]));
				std::vector<int> predictions = network.predict(eval_images);
				if(user_input_c > 4)
				{
					eval_labels = readmnistgz(std::string(user_input[4]), ".gz");
					for(int i = 0;i<eval_labels.size();i++)
					{
						std::cout << "Prediction: " << predictions[i] << " - Ground truth: " << eval_labels[i] << std::endl;
					}
				}
				else
				{
					for(int i = 0;i<eval_images.size()/;i++)
					{
						std::cout << "Prediction: " << predictions[i] << std::endl;
					}
				}
				std::cout << std::endl;
			}
		}
		//Read user input

	}
	return 0;
}
