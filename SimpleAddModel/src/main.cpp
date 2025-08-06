
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

int main()
{
	TF::MLModel model("simple_add");

	model.AddInput("x", 
				   TF::DataType::Float32,
				   { -1 });

	model.AddInput("y", 
				   TF::DataType::Float32,
				   { -1 });

	model.AddOutput("add_result");

	model.AddLayer(TF::LayerType::Add,
	{
		{ "input_names", { "x", "y" } },
		{ "output_name", "add_result" }
	});

	if (!model.CreateModel())
	{
		std::cerr << "Failed to Create Model." << std::endl;
		return -1;
	}

	// Example inputs
	cppflow::tensor input_x = cppflow::tensor({ 1.0f, 2.0f, 3.0f });
	cppflow::tensor input_y = cppflow::tensor({ 4.0f, 5.0f, 6.0f });

	std::unordered_map<std::string, cppflow::tensor> inputs;
	inputs["x"] = input_x;
	inputs["y"] = input_y;

	TF::MLModel::LabeledTensor results;
	if (model.Run(inputs, results))
	{
		// Output the results -------------------------------------------------------------------------
		std::cout << "Input X:\n" << TF::PrintTensor<float>(input_x) << std::endl;
		std::cout << "Input Y:\n" << TF::PrintTensor<float>(input_y) << std::endl;
		
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