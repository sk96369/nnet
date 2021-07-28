#include <vector>

//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
//possible integer being encoded and i is the number being encoded
std::vector<int> int_toOneHot(int i, int max)
{
	std::vector<int> vec(max+1, 0);
	vec[i] = 1;
	return vec;
}

int onehot_toInt(const std::vector<double> &oh) 
{
	int size = oh.size();
	int max = 0;
	for(int i = 0;i<size;i++)
	{
		if(oh[i] > oh[max])
			max = i;
	}
	return max;
}

int onehot_toInt(const std::vector<int> &oh) 
{
	int size = oh.size();
	int max = 0;
	for(int i = 0;i<size;i++)
	{
	if(oh[i] > oh[max])
		max = i;
	}
	return max;
}	
