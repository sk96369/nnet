#include "multiplym.h"
#include <sstream>
#include <vector>
#include <fstream>
#include <iterator>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <list>
#include <errno.h>
#include <stdio.h>
#include <tuple>
#include "onehot.cpp"
#include <float.h>

//multiplym.cpp
namespace MM
{
	

	NN::NN(int inputsize, int h1size, int outsize, std::string filename)
	{
		for(int i = 0;i<inputsize;i++)
		{
			input.push_back(0);
		}
		for(int i = 0;i<h1size;i++)
		{
			h1.push_back(0);
		}
		for(int i = 0;i<outsize;i++)
		{
			out.push_back(0);
		}

		std::ifstream file;
		file.open(filename);
		if(file)
		{
			size_t phase = 0;
			double val;
			while(phase < 10)
			{
				file >> val;
				std::cout << val << "\n";
				bias1.push_back(val);
				phase++;
			}
			while(phase < 20)
			{
				file >> val;
				bias2.push_back(val);
				phase++;
			}
			while(phase < 7840 + 20)
			{
				file >> val;
				wi.push_back(val);
				phase++;
			}
			while(file >> val)
			{
				w1.push_back(val);
				phase++;
			}
			file.close();
		}
		else
			std::cout << "File not found!\n";
	}			




    NN::NN(matrix<int> &in, int h1size, int outsize, int batch_size) 
    {
	
	input = in;
	h1 = matrix<double>(0.0, h1size, batch_size);
	bias1 = matrix<double>(0.01, 10, 1);
	bias2 = matrix<double>(0.01, 10, 1);
	out = matrix<double>(0.0, outsize, batch_size);
	wi = matrix<double>(-0.5, 0.5, 10, 10);
	w1 = matrix<double>(-0.5, 0.5, 10, 10);
        learningrate = 0.08;
    }

    

    void NN::fprop()
    {
        //Multiply the input layer with the weights between the input and h1 layers
	h1 = mm(input, wi);
	//Add the bias to each of the hidden states	
	add(h1, bias1);
        //Run relu function on the hidden states
        relu(h1);
	out = mm(h1, w1);
        softmax(out);
    }

    void NN::bprop(const matrix<int> &targetoutput)
    {
	int batch_size = targetoutput.rows();
        //The variables holding the info of how much the weights should be adjusted
        matrix<double> d_w1;
        matrix<double> d_inputweights;
        //The variables holding the info of how much the biases should be adjusted
        double dbias1;
        double dbias2;
        //Variable for the difference between target and generated output
        matrix<double> delta;
	matrix<double> delta2;
	

        //Calculate the difference between target and generated output
	delta = getError(targetoutput, out);
        //Calculate the adjustments needed for the weights and biases of the second layer
	d_w1 = scalar_m(mm(delta, transpose(h1)), 1/batch_size);
	dbias2 = scalar_m(sum_m(delta), 1/batch_size);

        //Calculate the adjustments needed for the weights and biases of the first layer
	delta2 = mm(transpose(w1), delta)
		lmultiply(lmultiply(t, w1), drelu(h1)); 

        d_inputweights = wmultiply(input, t2);

        dbias1 = bsum(t2);

       	updateParameters(d_inputweights, d_w1, dbias1, dbias2); 
    }

    void NN::updateParameters(const std::vector<double> &d_inputweights, const std::vector<double> &d_w1, double dbias1, double dbias2)
    {
        for(int i = 0;i<wi.size();i++)
        {
            wi[i] = wi[i] - learningrate * d_inputweights[i];
        }
        for(int i = 0;i<w1.size();i++)
        {
            w1[i] = w1[i] - learningrate * d_w1[i];
        }
        for(int i = 0;i<bias1.size();i++)
        {
		            bias1[i] -= learningrate * dbias1;
        }
	std::cout <<"dbias1: " << dbias1 << " dbias2: " << dbias2 << std::endl;
		std::cin.get();

        for(int i = 0;i<bias2.size();i++)
        {
            bias2[i] -= learningrate * dbias2;
        }
    }


        void train(std::vector<int> labels, int batch_size, int epoch)
	{
		int correct = 0;
		int wrong = 0;
		size_t interval = 0;
		size_t count = 0;
		for(int j = 0;j < epoch;j++)
		{
        		fprop();
			matrix<int> oh_labels = int_toOneHot(labels, 10);

			bprop(oh_labels);
			}
			else
			{
//				std::cout << "Correct output at: " << i << std::endl;
				i = 100;
				correct++;
				label_correct++;
			}
//			std::cin.get();
		}
		interval++;
		count++;
		if(interval == 100)
		{
			std::cout<< "Accuracy of the predictions for the last label: " << (double)label_correct / ((double)label_correct+(double)label_incorrect) << "\nOverall accuracy: " << ((double)correct/((double)correct+(double)wrong))*100 << "%\nTraining pairs used: " << count << std::endl;
			interval = 0;
			label_correct = 0;
			label_incorrect = 0;
		}
	}

    std::vector<double> NN::lmultiply(const std::vector<int> &left, const std::vector<double> &right) const
    {
        int lsize = left.size();
        int rsize = right.size();
        //osize is the number of "columns" in the weight matrix, sort of
        int osize = rsize/lsize;
        std::vector<double> output(osize);
	for(int i = 0;i < lsize;i++)
        {
            for(int j = 0;j < osize;j++)
            {
                output[j] += (double)left[i] * right[i*osize + j];
            }
        }
        return output;
    }


    std::vector<double> NN::lmultiply(const std::vector<double> &left, const std::vector<double> &right) const
    {
        int lsize = left.size();
        int rsize = right.size();
        //osize is the number of "columns" in the weight matrix, sort of
        int osize = rsize/lsize;
        std::vector<double> output(osize);
        for(int i = 0;i < lsize;i++)
        {
            for(int j = 0;j < osize;j++)
            {
                output[j] += left[i] * right[i*osize + j];
            }
        }
        return output;
    }

    std::vector<double> NN::wmultiply(const std::vector<double> &left, const std::vector<double> &right) const
    {
        std::vector<double> weights(left.size()*right.size(), 0);
        for(int i = 0;i<left.size();i++)
        {
            for(int j = 0;j<right.size();j++)
            {
                weights[i*10+j] = left[i] * right[j];
            }
        }
        return weights;
    }

    std::vector<double> NN::wmultiply(const std::vector<int> &left, const std::vector<double> &right) const
    {
        
        std::vector<double> weights(left.size()*right.size(), 0);
        for(int i = 0;i<left.size();i++)
        {
            for(int j = 0;j<right.size();j++)
            {
                weights[i*10+j] = left[i] * right[j];
            }
        }
        return weights;
    }

    double NN::bsum(const std::vector<double> &v) const
    {
        double d = 0;
        for(auto& i : v)
        {
            d += i;
        }
        return d;
    }
	
    bool NN::setInput(const std::vector<int> &in)
    {
	    if(in.size() == 28*28)
	    {
		    for(int i = 0;i < 28*28;i++)
		    {
			    input[i] = in[i];
		    }
	    }
	    else
		    return false;
	    return true;
    }

    bool NN::saveModel(std::string filename) const
    {
	    filename.append(".txt");
	std::ofstream file;
	file.open(filename);
	for(auto ptr = bias1.begin();ptr != bias1.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = bias2.begin();ptr != bias2.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = wi.begin();ptr != wi.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;

	for(auto ptr = w1.begin();ptr != w1.end();ptr++)
	{
		file << *ptr << " ";
	}
	file << std::endl;
	file.close();
	return 1;
    }

    int NN::predict(const std::vector<int> &in)
    {
	    setInput(in);
	    std::cout << "TEST2\n";
	    fprop();
	    std::cout << "TEST3\n";
	    return onehot_toInt(out);
    }


}
