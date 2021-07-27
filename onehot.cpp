#include <vector>

namespace OH
{
	//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
	//possible integer being encoded and i is the number being encoded
	std::vector<int> toOneHot(int i, int max)
	{
		std::vector<int> vec(max+1, 0);
		vec[i] = 1;
		return vec;
	}
}
		
