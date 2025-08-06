
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

int main()
{
	TF::MLModel model("linear");

	model.AddInput("x", 
				   TF::DataType::Float32,
				   { -1, 1 });

	model.AddOutput("y");

	model.AddLayer(TF::LayerType::Dense,
	{
		{ "input_name", "x" },
		{ "units", 1 },
		{ "output_name", "y" }
	});

	if (!model.CreateModel())
	{
		std::cerr << "Failed to Create Model." << std::endl;
		return -1;
	}

	// Example Inputs
	cppflow::tensor input_x = cppflow::tensor(std::vector<float>{ 1.0f, 2.0f, 3.0f }, { 3, 1 });

	std::unordered_map<std::string, cppflow::tensor> inputs;
	inputs["x"] = input_x;

	TF::MLModel::LabeledTensor results;
	if (model.Run(inputs, results))
	{
		// Output the results -------------------------------------------------------------------------
		std::cout << "Input X:\n" << TF::PrintTensor<float>(input_x) << std::endl;

		if (!results.empty())
		{
			for (const auto& [key, value] : results)
			{
				std::cout << "Result {" << key << "}:" << std::endl;
				std::cout << "\t" << TF::PrintTensor<float>(value) << std::endl;
			}
		}
		// --------------------------------------------------------------------------------------------
	}
	else
	{
		std::cerr << "Failed to Run Model." << std::endl;
		return -1;
	}
	return 0;
}