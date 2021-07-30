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

matrix<int> int_toOneHot(std::vector<int> &in, int max)
{
	int rows = in.size();
	//Creates a new matrix with all the elements set to 0
	matrix<int> onehot((int)0, max, rows);
	for(int i = 0;i<rows;i++)
	{
		//Sets the element at in[i] to 1
		onehot[i][in[i]] = 1;
	}
	return onehot;
}

std::vector<int> onehot_toInt(const matrix<double> &oh)
{
	int collumns = oh.collumns();
	int rows = oh.rows();
	std::vector<int> outputs_as_integers;
	for(int i = 0;i<rows;i++)
	{
		int max = 0;
		for(int j = 0;j<collumns;j++)
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
