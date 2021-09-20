#include <iostream>

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
