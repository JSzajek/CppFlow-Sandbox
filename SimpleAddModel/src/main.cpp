
#include <iostream>
#include <cstdlib>

#include <cppflow/cppflow.h>

#include "TFModel.h"

template<typename T>
std::string PrintTensor(const cppflow::tensor& tensor)
{
	std::stringstream stream;
	stream << "[";
	std::vector<T> data = tensor.get_data<T>();
	for (uint32_t j = 0; j < data.size(); ++j)
	{
		stream << data[j];
		if (j < data.size() - 1)
			stream << ", ";
	}
	stream << "]";
	return stream.str();
}


int main()
{
	const std::string model_name = "simple_add";

	// Create Simple Model Description ------------------------------------------------------------
	TFModelLayout layout;
	layout.model_name = model_name;

	layout.inputs = 
	{
		{ "x", "float32", {-1} },
		{ "y", "float32", {-1} }
	};

	layout.outputs = 
	{
		{ "add_result" }
	};

	layout.layers = 
	{
		{ "Add", 
			{
				{"input_names", {"x", "y"}}, 

				{"output_name", "add_result"}
			} 
		}
	};

	nlohmann::json j = layout.to_json();
	std::ofstream("model_description.json") << j.dump(2);
	// --------------------------------------------------------------------------------------------

	// Create the Model In Python -----------------------------------------------------------------
	const std::string python_script = "python ../PythonScripts/build_model_from_json.py \"model_description.json\"";

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

	// Example inputs (adjust to your actual inputs and data)
	cppflow::tensor input_x = cppflow::tensor({ 1.0f, 2.0f, 3.0f });
	cppflow::tensor input_y = cppflow::tensor({ 4.0f, 5.0f, 6.0f });

	// Match names from JSON to create input tuples
	if (io_names["inputs"].contains("x")) 
	{
		inputs_vec.emplace_back(io_names["inputs"]["x"].get<std::string>(), input_x);
	}
	if (io_names["inputs"].contains("y")) 
	{
		inputs_vec.emplace_back(io_names["inputs"]["y"].get<std::string>(), input_y);
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
	std::cout << "Input X:\n" << PrintTensor<float>(input_x) << std::endl;
	std::cout << "Input Y:\n" << PrintTensor<float>(input_y) << std::endl;
	
	if (!results.empty()) 
	{
		for (uint32_t i = 0; i < results.size(); ++i)
		{
			std::cout << "Result {" << i << "}:" << std::endl;
			std::cout << "\t" << PrintTensor<float>(results[i]) << std::endl;
		}
	}
	// --------------------------------------------------------------------------------------------

	return 0;
}