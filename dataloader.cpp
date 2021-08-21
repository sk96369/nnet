#include <vector>
#include <iostream>
#include <string>
#include "dataloader.h"
#include <sstream>
#include "gzreader.h"

std::vector<std::string> parseFilename(std::string filename_full)
{
	std::vector<std::string> parsed(2);
	std::stringstream ss;
	ss.str(filename_full);
	std::getline(ss, parsed[0], '.');
	if(ss.rdbuf()->in_avail() > -1)
	{
		parsed[1] = ".";
		std::string extension;
		std::getline(ss, extension);
		parsed[1].append(extension);
		std::cout << parsed[1] <<"\n";
	}
	else
		parsed[1] = "";
	return parsed;
}

std::vector<int> loadData(std::string str)
{
	std::vector<std::string> parsed = parseFilename(str);
	std::vector<int> data;
	if(parsed[1] != "")
	{
		if(parsed[1] == ".gz")
		{
			data = readmnistgz(parsed[0], parsed[1]);
		}
		else if(parsed[1] == ".txt")
		{
		}
		else if(parsed[1] == ".json")
		{
		}
		else
		{
			std::cout << "File type not recognized.\n";
		}
	}
	else
	{
		std::cout << "Invalid filename, please include the file extension (f.e. \"file.gz\")\n";
	}
	return data;
}
