#include <vector>
#include "onehot.h"
#include "mm_math.h"

//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
//possible integer being encoded and i is the number being encoded
std::vector<int> int_toOneHot(int i, int max)
{
	std::vector<int> vec(max+1, 0);
	vec[i] = 1;
	return vec;
}

//List of integers into a one-hot-encoded matrix
MM::mat<int> int_toOneHot(std::vector<int> &in, int max)
{
	int rows = in.size();
	//Creates a new matrix with all the elements set to 0
	MM::mat<int> onehot((int)0, max, rows);
	for(int i = 0;i<rows;i++)
	{
		//Sets the element at in[i] to 1
		onehot.m[i][in[i]] = 1;
	}
	return onehot;
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

std::vector<int> onehot_toInt(const MM::mat<double> &oh)
{
	int columns = oh.columns();
	int rows = oh.rows();
	std::vector<int> outputs_as_integers;
	for(int i = 0;i<rows;i++)
	{
		int max = 0;
		for(int j = 0;j<columns;j++)
		{
			if(oh.m[i][j] > max)
			{
				max = oh.m[i][j];
			}
		}
		outputs_as_integers.push_back(max);
	}
	return outputs_as_integers;
}
