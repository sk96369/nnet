#include <fstream>
#include <iostream>
#include <string>
#include "gzreader.h"
#include <cstring>
#include "mm_math.h"
#include "multiplym.h"
#include <sstream>

const int datasize = 60000;
const int batchsize = 3000;
const int epoch = 3;

const std::string training_labels_filename = "traininglabels";
const std::string training_images_filename = "imagedata";
const std::string eval_labels_filename = "eval_labels";
const std::string eval_images_filename = "eval_images";

//Main.cpp
int main(int argc, char* argv[])
{
    std::fstream pretrained_file;

//	std::list<std::tuple<std::vector<int>, int>> trainingdata;

    if(strcmp(argv[1], "test") ==0)
    {
	    if(argc < 3)
	    {
		    std::cout << "the \"test\"-command requires another argument to point to the model file.\nExample: \"./a.out test model\" points to a file called \"model.txt\".\n";
		    return 1;
	    }
	std::vector<int> labels = readmnistgz(eval_labels_filename);
	std::vector<int> images = readmnistgz(eval_images_filename);
	std::vector<int> labels_batch;
	std::vector<int> images_batch;
	MM::mat<int>imagematrix(9, 784, batchsize);
	MM::nnet network (imagematrix, 10, 10, batchsize);
	network.setParameters(argv[2]);

	int iterations = datasize/batchsize;
	for(int n = 0;n < iterations;n++)
	{
		for(int i = 0;i < batchsize;i++)
		{
			labels_batch.push_back(labels[n*batchsize + i]);
		}
		for(int i = 0;i < batchsize*784;i++)
		{
			images_batch.push_back(images[n*batchsize*784 + i]);
		}
		
		imagematrix.newValues(images_batch, 784, batchsize);
		network.setInput(images_batch, 784, batchsize);
		network.predict(labels_batch);
	
		//Run the output matrix through the onehot_toint function to get a list of predictions
		//in integer form
		std::vector<int> predictions = network.getPredictions();
		for(int i = 0;i<batchsize;i++)
		{
			std::cout << "Prediction: " << predictions[i] << " - Expected output: " << labels_batch[i] << "\n";
		}
		labels_batch.clear();
		images_batch.clear();
	}
    }
    if(strcmp(argv[1], "loadparams") == 0)
    {
        if(argc != 3)
        {
            std::cout << "Incorrect number of arguments";
            exit(1);
        }
	std::string str;
	std::cout << "Type in the filename to make predictions on\n";
	std::cin >> str;
	std::ifstream numberfile;
	numberfile.open(str);
	std::cout << "TEST\n";
	char charpixel;
	if(numberfile)
	{
		std::vector<int> image;

		while(numberfile.get(charpixel))
		{
			if(charpixel != '\n')
				image.push_back((int)charpixel);
		}
		std::cout << "Image size: " << image.size() << " pixels.\n";
	//	std::cout << "Prediction: " << network.predict(image) << std::endl;
		image.clear();
		numberfile.close();
	}
	else
	{
		std::cout << "File does not exist!";
		exit(1);
	}

	}

	if(strcmp(argv[1], "trainmodel") == 0)
	{
		std::vector<int> labels = readmnistgz(training_labels_filename);
		std::vector<int> images = readmnistgz(training_images_filename);
		std::vector<int> labels_batch;
		std::vector<int> images_batch;
		MM::mat<int>imagematrix(9, 784, batchsize);

		MM::nnet network(imagematrix, 10, 10, batchsize);
		int iterations = datasize/batchsize;
		
		for(int n = 0;n <= iterations;n++)
		{
			for(int i = 0;i < batchsize;i++)
			{
				labels_batch.push_back(labels[n*batchsize + i]);
			}
			for(int i = 0;i < batchsize*784;i++)
			{
				images_batch.push_back(images[n*batchsize*784 + i]);
			}
			
			imagematrix.newValues(images_batch, 784, batchsize);
			network.setInput(images_batch, 784, batchsize);
	
			network.train(labels_batch, batchsize, epoch);
			labels_batch.clear();
			images_batch.clear();
		}	
		network.saveModel(argv[2]);
	}
		

	return 0;
}
