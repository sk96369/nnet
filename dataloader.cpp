#include <vector>
#include <iostream>
#include <string>
#include "dataloader.h"
#include <sstream>
#include "gzreader.h"
#include <filesystem>
#include <fstream>

std::vector<std::string> listInputFormats(int version)
{
	std::vector<std::string> inputFormats;
	if(version > 0)
	{
		inputFormats.push_back(".gz");
		inputFormats.push_back(".txt");
	}
}

std::vector<std::string> listFiles(std::string path, const std::vector<std::string> &extensions)
{
	std::vector<std::string> filenames;
	std::filesystem::create_directories(path);
	//Iterate through all the files in the directory path
	for(auto& ptr : std::filesystem::directory_iterator(path))
	{
		std::vector<std::string> help = &ptr.split('.');
		//Check that the file has a matching extension 
		for(auto& extension : extensions)
		{
			if(help.size() == 2 && help[1] == extension)
			{
				filenames.push_back(&ptr);
			}
		}
	}
	return filenames;
}

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

bool getFile(const std::string &path, std::string &selectedFile)
{
	std::vector<std::string> filenames = listFiles(path, listInputFormats(VERSION));
	if(filenames.size() > 0)
	{
		int selection = makeSelection(filenames, 3);
		if(!readInt(std::cin, selection))
		{
			std::cout << "Invalid input!\n";
			return false;
		}
		else
		{
			selectedFile = filenames[selection];
			return true;
		}
	}
	//If the program executes to this point, no files have been found and false is returned
	std::cout << "No files found in directory " << path << "/\n";
	return false;
}

std::vector<int> loadData(std::string filename, std::vector<int> &metadata, std::vector<int> &output)
{
	std::vector<std::string> parsed = parseFilename(filename);
	if(parsed[1] != "")
	{
		if(parsed[1] == ".gz")
		{
			output = readmnistgz(parsed[0], parsed[1], metadata);
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
	return output;
}
