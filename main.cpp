#include "zlib.h"
#include "multiplym.h"
#include <fstream>
#include <iostream>
#include <string>
#include "gzreader.cpp"

//Main.cpp
int main(int argc, char* argv[])
{
    std::ofstream parameters;
    std::fstream pretrained_file;
    std::string trainingdata_filename = "traininglabels";
    std::string trainingdata2_filename = "imagedata";
	std::list<std::tuple<std::vector<int>, int>> trainingdata;
    if(strcmp(argv[1], "loadparams") == 0)
    {
        pretrained_file.open("model.txt");
        if(!parameters)
        {
            std::cout << "Unable to open \"model.txt\"";
            exit(1);
        }
        std::string line;
        while(pretrained_file >> line)
        {
            std::cout << line << std::endl;
        }
	pretrained_file.close();
    }

    if(strcmp(argv[1], "trainmodel") == 0)
    {
	    std::vector<int> labels = readmnistgz(trainingdata_filename);
	    std::vector<int> images = readmnistgz(trainingdata2_filename);

	    for(int i = 0;i < labels.size();i++)
	    {
		    std::vector<int> image;
		    for(int j = 0;j < 784;j++)
		    {
			    image.push_back(images[i*784+j]);
		    }
		    std::tuple<std::vector<int>, int> nextmember = std::make_tuple(image, labels[i]);
		    trainingdata.push_back(nextmember);
	    }
	    MM::NN network(10, 10);
	    network.train(trainingdata);

	    network.saveModel("model.txt");
    }

    return 0;
}
