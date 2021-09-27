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
		ModelSettings() : activation(RELU), printwidth(1), trainingInputFile(""), trainingLabelFile(""), evalInputFile(""), evalLabelFile("") {}
		ModelSettings(int a, int pw) : printwidth(pw), activation(a) {}
		void edit();
		std::string trainingInputFile;
		std::string trainingLabelFile;
		std::string evalInputFile;
		std::string evalLabelFile;
		
		// Getters for the string forms of the settings
		std::string getTrainingInputFile();
		std::string getTrainingLabelFile();
		std::string getEvalInputFile();
		std::string getEvalLabelFile();
	};

	struct ProgramSettings
	{
		//Default settings for new models. Copied to new model whenever one is created
		ModelSettings defaultModelSettings;
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
