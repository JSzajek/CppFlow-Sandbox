
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

int main()
{
	TF::MLModel model("linear");

	model.AddInput("x", 
				   TF::DataType::Float32, 
				   { -1, 4, 1 });

	model.AddOutput("y");

	model.AddLayer("Flatten",
	{
		{ "input_name", "x" },
		{ "output_name", "flat_input" }
	});

	model.AddLayer("Dense",
	{
		{ "input_name", "flat_input" },
		{ "units", 16 },
		{ "output_name", "dense_1" },
	});

	model.AddLayer("Dense",
	{
		{ "input_name", "dense_1" },
		{ "units", 2 },
		{ "output_name", "y" },
	});

	if (!model.CreateModel())
	{
		std::cerr << "Failed to create model." << std::endl;
		return -1;
	}


	// Example Input
	cppflow::tensor input_x = cppflow::tensor(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f }, { 4, 1 });

	std::unordered_map<std::string, cppflow::tensor> inputs;
	inputs["x"] = input_x;

	std::cout << "Input X:\n" << TF::PrintTensor<float>(input_x) << std::endl;


	// Output the Pre-Training Results
	std::cout << "Pre-Training Output" << std::endl;
	TF::MLModel::Result pre_results;
	if (model.Run(inputs, pre_results))
	{
		if (!pre_results.empty())
		{
			for (const auto& [key, value] : pre_results)
			{
				std::cout << "Result {" << key << "}:" << std::endl;
				std::cout << "\t" << TF::PrintTensor<float>(value) << std::endl;
			}
		}
	}
	else
	{
		std::cerr << "Failed to Run Model." << std::endl;
		return -1;
	}


	// Add Training Data
	model.AddTrainingData("x", 
						  { 5.1f, 3.5f, 1.4f, 0.2f }, 
						  "y", 
						  { 1.0f, 0.0f });

	model.AddTrainingData("x", 
						  { 6.2f, 3.4f, 5.4f, 2.3f }, 
						  "y", 
						  { 0, 0.5f });

	model.AddTrainingData("x", 
						  { 0.2f, 6.8f, 9.1f, 1.2f }, 
						  "y", 
						  { 0, 1.0f });

	if (!model.TrainModel(64))
	{
		std::cerr << "Failed to Train Model." << std::endl;
		return -1;
	}

	// Output the Post-Training Results
	TF::MLModel::Result post_results;
	if (model.Run(inputs, post_results))
	{
		if (!post_results.empty())
		{
			for (const auto& [key, value] : post_results)
			{
				std::cout << "Result {" << key << "}:" << std::endl;
				std::cout << "\t" << TF::PrintTensor<float>(value) << std::endl;
			}
		}
	}
	else
	{
		std::cerr << "Failed to Run Model." << std::endl;
		return -1;
	}
	return 0;
}