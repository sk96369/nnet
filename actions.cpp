#include <iostream>
#include "dataloader.h"
#define VERSION 1

int confirm()
{
	std::cout << "Are you sure? (y/n)\n";
	std::string a;
	std::cin >> a;
	if(a == "y" || a == "Y")
	{
		std::cout << "Yes\n";
		return 1;
	}
	std::cout << "No\n";
	return 0;
}

int makeSelection(const std::vector<std::string> &options, int indent = 0)
{
	for(int i = 0;i<options.size();i++)
	{
		std::string indent(indent, ' ');
		std::cout << indent <<  i << ". " << options[i] << std::endl;
	}
	int index;
	while(!readInt(std::cin, index))
		return index;
}
