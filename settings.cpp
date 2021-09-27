

namespace MM
{
	std::vector<std::string> modelSettings::getSettings()
	{
		printf("1. Activation function: %s\n
				2. Print width: %s\n
				3. Training input file: %s\n
				4. Training label file: %s\n
				5. Evaluation input file: %s\n
				6. Evaluation label file: %s\n",
				getActivationFunction(),
				getPrintWidth(),
				getTrainingInputFile(),
				getTrainingLabelFile(),
				getEvalInputFile(),
				getEvalLabelFile());

	}

	void modelSettings::edit()
	{
		bool done = false;
		while(!done)
		{
			std::vector<std::string> settingsList = getSettings();



	}

	//Getters for the string forms of the settings
	std::string getActivationFunction()
	{
		std::string activationString;
		switch(activation)
		{
			case RELU :
				activationString = "ReLU";
				break;
			case LEAKY_RELU :
				activationString = "Leaky ReLU";
				break;
			default:
				activationString = "None";
				break;
		}
		return activationString;
	}

	std::string getPrintWidth()
	{
		return std::to_string(printWidth);
	}

	std::string getTrainingInputFile()
	{
		if(trainingInputFile == "")
			return "None";
		return trainingInputFile;
	}

	std::string getTrainingLabelFile()
	{
		if(trainingLabelFile == "")
			return "None";
		return trainingLabelFile;
	}

	std::string getEvalInputFile()
	{
		if(evalInputFile == "")
			return "None";
		return evalInputFile;
	}

	std::string getEvalLabelFile()
	{
		if(evalLabelFile == "")
			return "None";
		return evalLabelFile;
	}
}
