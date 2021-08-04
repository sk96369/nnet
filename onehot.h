#pragma once

#include "headers.h"
#include <vector>

//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
//possible integer being encoded and i is the number being encoded
std::vector<int> int_toOneHot(int i, int max);

int single_onehot_toInt(const std::vector<int> &oh);

int single_onehot_toInt(const std::vector<double> &oh);

//List of integers into a one-hot-encoded matrix
MM::mat<int> int_toOneHot(const std::vector<int> &in, int max);

std::vector<int> onehot_toInt(MM::mat<double> &oh);
