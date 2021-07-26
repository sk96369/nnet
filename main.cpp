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
    const char * trainingdata_filename = "traininglabels.gz";
    const char * trainingdata2_filename = "imagedata.gz";
	std::list<std::tuple<std::vector<int>, std::vector<int>>> trainingdata;
    if(argv[1] == "loadparams")
    {
        parameters.open(params);
        if(!parameters)
        {
            std::cout << "Unable to open \"model.txt\"";
            exit(1);
        }
        std::string line;
        while(parameters >> line)
        {
            std::cout << line << std::endl;
        }
    }

    if(argv[1] == "trainmodel")
    {
	trainingdata = toTrainingData(trainingdata_filename, trainingdata2_filename);
    }

    MM::NN network(10, 10);
	network.train(trainingdata);

	parameters.open("model.txt");

    return 0;
}
