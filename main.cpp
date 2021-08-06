#include <fstream>
#include <iostream>
#include <string>
#include "gzreader.h"
#include <cstring>
#include "mm_math.h"
#include "multiplym.h"

const int datasize = 60000;
const int batchsize = 10;
const int epoch = 1;

//Main.cpp
int main(int argc, char* argv[])
{
    std::fstream pretrained_file;
    std::string trainingdata_filename = "traininglabels";
    std::string trainingdata2_filename = "imagedata";
//	std::list<std::tuple<std::vector<int>, int>> trainingdata;
    if(strcmp(argv[1], "loadparams") == 0)
    {
        if(argc != 3)
        {
            std::cout << "Incorrect number of arguments";
            exit(1);
        }
	//MM::nnet network(784, 10, 10, argv[2]);
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
		std::vector<int> labels = readmnistgz(trainingdata_filename);
		std::vector<int> images = readmnistgz(trainingdata2_filename);
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
			for(int i = 0;i < batchsize;i++)
			{
				for(int j = 0;j<784;j++)
				{
					images_batch.push_back(images[n*batchsize + i*784 + j]);
				}
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
