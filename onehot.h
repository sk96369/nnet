#pragma once

#include <vector>
#include "mm_math.h"

//Constructs a one-hot-vector from 0 to max of size max+1, where max is the highest
//possible integer being encoded and i is the number being encoded
std::vector<int> int_toOneHot(int i, int max);

int onehot_toInt(const std::vector<int> &oh);

int onehot_toInt(const std::vector<double> &oh);

//List of integers into a one-hot-encoded matrix
MM::mat<int> int_toOneHot(std::vector<int> &in, int max);

std::vector<int> onehot_toInt(const MM::mat<double> &oh);
