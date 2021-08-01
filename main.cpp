#include "zlib.h"
#include "multiplym.h"
#include <fstream>
#include <iostream>
#include <string>
#include "gzreader.cpp"
#include "mm_math.h"

const int batchsize = 60000;
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
	//MM::NN network(784, 10, 10, argv[2]);
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
		MM::mat<int> imagematrix(images, 784, 60000);

	
		MM::NN network(imagematrix, 10, 10, batchsize);
	
		network.train(labels, batchsize, epoch);
	
		network.saveModel(argv[2]);
	}
		

	return 0;
}
