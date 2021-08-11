#ifndef ONEHOT_H
#define ONEHOT_H

#include <vector>
#include "mm_math.h"

//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
//possible integer being encoded and i is the number being encoded
std::vector<int> int_toOneHot(int i, int max)
{
	std::vector<int> vec(max+1, 0);
	vec[i] = 1;
	return vec;
}


int single_onehot_toInt(const std::vector<double> &oh)
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

int single_onehot_toInt(const std::vector<int> &oh)
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

//List of integers into a one-hot-encoded matrix
template<typename B>
MM::mat<int> int_toOneHot(const std::vector<B> &in, int max)
{
	int rows = in.size();
	//Creates a new matrix with all the elements set to 0
	MM::mat<int> onehot((int)0, max, rows);
	for(int i = 0;i<rows;i++)
	{
		//Sets the element at in[i] to 1
		onehot.m[i][(int)in[i]] = 1;
	}
	return onehot;
}

//List of integers into a one-hot-encoded double matrix
MM::mat<double> int_toOneHot(const std::vector<int> &in, int max)
{
	int rows = in.size();
	//Creates a new matrix with all the elements set to 0
	MM::mat<double> onehot((int)0, max, rows);
	for(int i = 0;i<rows;i++)
	{
		//Sets the element at in[i] to 1
		onehot.m[i][(int)in[i]] = 1;
	}
	return onehot;
}

template<typename B>
std::vector<int> onehot_toInt(const MM::mat<B> &oh)
{
	int columns = oh.columns();
	int rows = oh.rows();
	std::vector<int> outputs_as_integers;
	for(int i = 0;i<rows;i++)
	{
		int max = 0;
		for(int j = 0;j<columns;j++)
		{
			if(oh.m[i][j] > oh.m[i][max])
			{
				max = j;
			}
		}
		outputs_as_integers.push_back(max);
	}
	return outputs_as_integers;
}
#endif
