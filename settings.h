namespace MM
{
	enum Activation
	{
		RELU,
		LEAKY_RELU,
	}
	
	enum onoff
	{
		OFF,
		ON
	}

	enum Metadata
	{
		//Number of items
		ITEMS,
		ROWS,
		COLUMNS
	}

	class Profile
	{
		std::string name;
		ProgramSettings settings;
		int id;
		int version;
	};
	
	struct ModelSettings
	{
		//Activation function
		int activation;
		int printwidth;
		ModelSettings() : activation(RELU), printwidth(1) {}
		ModelSettings(int a, int pw);
		void edit();
	};

	struct ProgramSettings
	{
		int activation;
		int gpuMultithreading;
		double learningrate;
		//Filename for output, "" for standard output
		std::string outputfile;
		ProgramSettings() : gpuMultithreading(ON), learningrate(0.1), outputfile("")
		{
		}
		Settings(int a, int gmt, double lr, int pw, std::string of) : activation(a), gpuMultithreading(gmt), learningrate(lr), printwidth(pw), outputfile(of)
		{
		}
		void edit();
		std::string getOutputfile(){return outputfile;}
		void setOutputfile(std::string s){outputfile = s;}

	};
}
