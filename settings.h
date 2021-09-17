namespace MM
{
	enum Activation
	{
		RELU,
		LEAKY_RELU,
	}
	
	enum Device
	{
		OFF,
		ON
	}

	struct Settings
	{
		//Activation function
		int activation;
		int gpuMultithreading;
		double learningrate;
		//The number of elements printed on each row when printing the values of the matrices
		int printwidth;
		Settings() : activation(RELU), gpuMultithreading(ON), learningrate(0.1), printwidth(-1)
		{
		}
		Settings(int a, int gmt, double lr, int pw) : activation(a), gpuMultithreading(gmt), learningrate(lr), printwidth(pw)
		{
		}
	};
}
