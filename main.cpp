#include "zlib.h"
#include "multiplym.h"
#include <fstream>
#include <iostream>
#include <string>


//Main.cpp
int main(int argc, char* argv[])
{
    std::ifstream parameters;
    const char * trainingdata_filename = "traininglabels.gz";
    const char * params = "model.txt";
    struct gzFile_s *file;
    std::size_t wIdx = 0;
    FILE *param;

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


    if(argc != 2)
    {
        std::cout << " Usage : read_gzip [in gz File]" << std::endl;
        exit(1);
    }
    file = gzopen(argv[1], "rb");
    if(!file)
    {
        std::cout << " Failed to open gz file : " << argv[1] << std::endl;
        exit(1);
    }
    if(gzbuffer(file, 8192))
    {
        std::cout << " Failed to buffer the file " << argv[1] << std::endl;
        exit(1);
    }

    

    std::vector<double> inputs;
    double input;
    while(parameters >> input)
    {
        inputs.push_back(input);
    }
    parameters.close();

    MM::NN network(inputs, 10, 10);

    return 0;
}