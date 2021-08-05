#include "mm_math.h"

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

int main()
{
	std::vector<int> intlist;
	for(int i = 0;i<10;i++)
		intlist.push_back(i);
	MM::mat<int> onehot_ints = int_toOneHot(intlist, 10);


	return 0;
}
