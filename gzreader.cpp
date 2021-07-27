//CODE FROM THE INTERNET
#include <fstream>
#include <iostream>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <sstream>
#include <vector>

using namespace std;

vector<int> gzToint(int argc, char** argv)
{
	std::vector<int> inint;
	if(argc < 2)
	{
		cerr << "Usage: " << argv[0] << " <gzipped input file>" << endl;
	}
	//Read from the first command line argument, assume it's gzipped
	ifstream file(argv[1], ios_base::in | ios_base::binary);
	boost::iostreams::filtering_streambuf<boost::iostreams::input> inbuf;
    	inbuf.push(boost::iostreams::gzip_decompressor());
    	inbuf.push(file);
    	//Convert streambuf to istream
    	istream instream(&inbuf);
    	//Copy everything from instream to 
	stringstream ss;
    	int nextint;
	char nextchar;
	while(instream.get(nextchar))
	{
		nextint = (int)nextchar;
		inint.push_back(nextint);
	}

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
