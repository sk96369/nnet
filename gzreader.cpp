#include <iostream>
#include <fstream>
#include <string>
#include <zlib.h>
#include <cstring>
#include <stringstream>
#include <list>


std::list<std::tuple<std::vector<int>, std::vector<int>>> toTrainingData(std::string labelfilename, std::string imagefilename)
{
	//See: zlib.h
	std::size_t wIdx = 0;
	struct gzFile_s *gzFile;

	// IF THERE ARE PROBLEMS RUNNING IN WINDOWS, CHECK "zconf.h", search "ZEXTERN"
	gzFile = gzopen(labelfilename, "rb");
	if(gzFile)
	{
		if(gzbuffer(gzFile, 8192))
		{
			std::cout << "SOMETHING WENT WRONG." << std::endl;
			exit(1);
		}
	}

	int mn;
	int labelcount;
	unsigned char *data_p = new unsigned char[8192];
	memset(data_p, '\0', sizeof());

	wIdx = gzread(gzFile, voidp(data_p), 8192);
	std::vector<int> labels;
	std::stringstream ss;

	mn = data_p[0];
	labelcount = data_p[4];
	unsigned char label;

	for(int i = 8;i < data_p.size();i++)
	{
		labels.push_back(static_cast<int>(data_p[i]));
	}

	while(!gzeof(gzFile))
	{
		wIdx = gzread(gzFile, voidp(data_p), 8192);
		ss << data_p;
		while(data_p >> label)
		{
			labels.push_back(label);
		}
	}
	gzclose(gzFile);	

	//Do the same procedure for the image file
	gzFile = gzopen(imagefilename, "rb");
	if(gzFile)
	{
		if(gzbuffer(gzFile, 8192))
		{
			std::cout << "SOMETHING WENT WRONG." << std::endl;
			exit(1);
		}
	}

		
	wIdx = gzread(gzFile, voidp(data_p), 8192);
	std::vector<int> pixels;
	std::vector<int> images;

	mn = data_p[0];
	int numberofimages = data_p[4];
	int numberofrows = data_p[8];
	int numberofcolumns = data_p[12];
	char pixel;

	for(int i = 16;i < data_p.size();i++)
	{
		pixels.push_back(static_cast<int>(data_p[i]));
	}

	while(!gzeof(gzFile))
	{
		wIdx = gzread(gzFile, voidp(data_p), 8192);
		ss << data_p;
		while(data_p >> pixel)
		{
			pixels.push_back(static_cast<int>(pixel));
		}
	}
	gzclose(gzFile);

	//Make the appropriate containers for the training data
	std::list<std::tuple<std::vector<int>, std::vector<int>>> out;
	for(int i = 0;i < 60000;i++)
	{
		std::vector<int> image;
		for(int j = 0;j < 784;j++)
		{
			image.push_back(pixels[i*784+j]);
		}
		out.push_back(make_tuple(image, labels[i]));
	}

	return out;
}
