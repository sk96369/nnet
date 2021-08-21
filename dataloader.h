#include <vector>
#include <string>

//Splits the given string into a filename and its extension
std::vector<std::string> parseFilename(std::string filename_full);

//Reads data from a file with the given name, then creates and returns a vector of integers based on the file
std::vector<int> loadData(std::string filename_full);
