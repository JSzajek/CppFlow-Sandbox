
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

int main()
{
	const std::string model_name = "linear";
	const std::string model_description_path = model_name + "/model_description.json";

	// Create Model Description -------------------------------------------------------------------
	TF::ModelLayout layout;
	layout.model_name = model_name;

	layout.inputs = 
	{
		{ "x", "float32", { -1, 1 } },
	};

	layout.outputs = 
	{
		{ "y" }
	};

	layout.layers = 
	{
		{ "Dense", 
			{
				{ "input_name", "x" },
				{ "units", 1 }, 
				{ "output_name", "y" },
			} 
		}
	};

	layout.WriteToFile(model_description_path);
	// --------------------------------------------------------------------------------------------

	// Create the Model In Python -----------------------------------------------------------------
	const std::string python_script = "python ../PythonScripts/build_model_from_json.py \"" + model_description_path + "\"";

	int32_t exit_code = std::system(python_script.c_str());
	if (exit_code != 0)
	{
		std::cerr << "Failed to execute Python script. Exit code: " << exit_code << std::endl;
		return -1;
	}
	// --------------------------------------------------------------------------------------------

	// Load and Run The Model In C++ --------------------------------------------------------------

	// Load JSON with input/output tensor names
	std::ifstream in(model_name + "/cppflow_io_names.json");
	if (!in.is_open()) 
	{
		std::cerr << "Failed to open cppflow_io_names.json" << std::endl;
		return -1;
	}

	nlohmann::json io_names;
	in >> io_names;

	// Prepare inputs vector<std::tuple<string, tensor>>
	std::vector<std::tuple<std::string, cppflow::tensor>> inputs_vec;

	// Example inputs
	cppflow::tensor input_x = cppflow::tensor(std::vector<float>{ 1.0f, 2.0f, 3.0f }, { 3, 1 });

	// Match names from JSON to create input tuples
	if (io_names["inputs"].contains("x")) 
	{
		inputs_vec.emplace_back(io_names["inputs"]["x"].get<std::string>(), input_x);
	}

	// Prepare outputs vector<string>
	std::vector<std::string> outputs_vec;
	for (auto& [key, val] : io_names["outputs"].items()) 
	{
		outputs_vec.push_back(val.get<std::string>());
	}

	// Load model
	cppflow::model model(model_name);

	// Run model
	std::vector<cppflow::tensor> results = model(inputs_vec, outputs_vec);
	// --------------------------------------------------------------------------------------------


	// Output the results -------------------------------------------------------------------------
	std::cout << "Input X:\n" << TF::PrintTensor<float>(input_x) << std::endl;
	
	if (!results.empty()) 
	{
		for (uint32_t i = 0; i < results.size(); ++i)
		{
			std::cout << "Result {" << i << "}:" << std::endl;
			std::cout << "\t" << TF::PrintTensor<float>(results[i]) << std::endl;
		}
	}
	// --------------------------------------------------------------------------------------------

	return 0;
}