//CODE FROM THE INTERNET
#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <vector>
#include "gzreader.h"
#include "settings.h"

using namespace std;


//Function for reading the MNIST training data, and returning it in a vector
vector<int> readmnistgz(string filename, string extension, std::vector<int> &metadata)
{
	filename.append(extension);
	std::vector<int> inint;
	if(!boost::filesystem::exists(filename))
	{
		std::cout << "No file named \n" << filename << "\" can be found!\n";
		return inint;
	}
	const char progress[4] = {'-', '\\', '|', '/'};
	
	int numberofitems;
	int sizeofitems = 1;

	//Read from the file with the name given as the first parameter
	ifstream file(filename, ios_base::in | ios_base::binary);
	boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
    	inbuf.push(boost::iostreams::gzip_decompressor());
    	inbuf.push(file);
    	//Convert streambuf to istream
    	istream instream(&inbuf);
    	//Copy everything from instream to 
	stringstream ss;
    	int nextint;
	char nextchar;

	//Read the metadata
	std::string intsizechar(4, 0);

	cout << "Magic number: " ;
	for(size_t i = 0;i<sizeof(int);i++)
	{
		instream.get(nextchar);
		intsizechar[i] = nextchar;
	}
	nextint = (int)(((unsigned char)intsizechar[0]) << 24)
			| (int)(((unsigned char)intsizechar[1]) << 16)
			| (int)(((unsigned char)intsizechar[2]) << 8)
			| (int)(((unsigned char)intsizechar[3]));
	
	std::cout << nextint << std::endl;

	//If nextint == 2049, the data being read is the labels
	if(nextint == 2049)
	{
		cout << "Number of items: " ;
		for(size_t i = 0;i<sizeof(int);i++)
		{
			instream.get(nextchar);
			intsizechar[i] = nextchar;
		}
		nextint = (int)(((unsigned char)intsizechar[0]) << 24)
				| (int)(((unsigned char)intsizechar[1]) << 16)
				| (int)(((unsigned char)intsizechar[2]) << 8)
				| (int)(((unsigned char)intsizechar[3]));
		
		numberofitems = nextint;
		metadata[MM::ITEMS] = numberofitems;

		std::cout << nextint << std::endl;
	}
	//Assume we are processing the training images data
	else
	{
		sizeofitems = 784;
		cout << "Number of images: " ;
		for(size_t i = 0;i<sizeof(int);i++)
		{
			instream.get(nextchar);
			intsizechar[i] = nextchar;
		}
		nextint = (int)(((unsigned char)intsizechar[0]) << 24)
				| (int)(((unsigned char)intsizechar[1]) << 16)
				| (int)(((unsigned char)intsizechar[2]) << 8)
				| (int)(((unsigned char)intsizechar[3]));
		
		numberofitems = nextint;
		metadata[MM::ITEMS] = numberofitems;
		std::cout << nextint << std::endl;

		cout << "Number of rows: " ;
		for(size_t i = 0;i<sizeof(int);i++)
		{
			instream.get(nextchar);
			intsizechar[i] = nextchar;
		}
		nextint = (int)(((unsigned char)intsizechar[0]) << 24)
				| (int)(((unsigned char)intsizechar[1]) << 16)
				| (int)(((unsigned char)intsizechar[2]) << 8)
				| (int)(((unsigned char)intsizechar[3]));
		metadata[MM::ROWS] = nextint;
		std::cout << nextint << std::endl;

		cout << "Number of columns: " ;
		for(size_t i = 0;i<sizeof(int);i++)
		{
			instream.get(nextchar);
			intsizechar[i] = nextchar;
		}
		nextint = (unsigned int)(((unsigned char)intsizechar[0]) << 24)
				| (int)(((unsigned char)intsizechar[1]) << 16)
				| (int)(((unsigned char)intsizechar[2]) << 8)
				| (int)(((unsigned char)intsizechar[3]));
		metadata[MM::COLUMNS] = nextint;
		std::cout << nextint << std::endl;
	}
	unsigned char* buffer = new unsigned char[numberofitems*sizeofitems];
	instream.read((char*)buffer, numberofitems*sizeofitems);
	for(int i = 0;i<numberofitems*sizeofitems;i++)
	{
		int rotation = 0;
		if(i%5000 == 0)
		{
			std::cout << "Reading data from file " << filename << " " << progress[rotation % 4] << "\r";
			fflush(stdout);
			rotation++;
		}
		inint.push_back(static_cast<int>(buffer[i]));
	}
	std::cout << "Data read from file " << filename << ".\n";
	delete[] buffer;
    	//Cleanup
    	file.close();
	return inint;

}	
	


//OWN CODE
/*#include "gzreader.cpp"
#include <iostream>
#include <zlib.h>
#include <fstream>
#include <sstream>

int main()
{
	//Uncomment this to test for real
	//toTrainingData("traininglabels.gz", "imagedata.gz");
	
	std::size_t wIdx = 0;
	struct gzFile_s *gzFile;
	gzFile = gzopen("traininglabels.gz", "rb");
	if(gzFile)
	{
		if(gzbuffer(gzFile, 8192))
		{
			std::cout << "SOMETHING WENT WRONG.\n";
		}
	}

	unsigned char* data_p = new unsigned char[8192];
	memset(data_p, '\0', sizeof(*data_p));


	while(wIdx = gzread(gzFile, voidp(data_p), 8192))
	{
		std::cout << wIdx << std::endl;
		std::stringstream ss;
		ss << data_p;
		std::cout << ss.str() << std::endl;

	}

	gzclose(gzFile);

	
	return 0;
}*/

//Main function for testing
/*
int main()
{
	vector<int> labels = readmnistgz("traininglabels");
	vector<int> images = readmnistgz("imagedata");
	for(int i = 0;i < labels.size();i++)
	{
		printf("\nCorrect label: %i", labels[i]);
		for(int j = 0;j < 784;j++)
		{
			if(j % 28 == 0)
				printf("\n");
			printf("%i", images[i*784+j] != 0);
					}
	}
	printf("\n");
}
*/
