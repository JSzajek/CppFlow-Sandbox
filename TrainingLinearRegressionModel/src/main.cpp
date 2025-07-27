
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

int main()
{
	const std::string model_name = "linear";
	const std::string model_description_path = model_name + "/model_description.json";

	const std::string trained_model_name = "trained_linear";

	// Create Model Description -------------------------------------------------------------------
	TF::ModelLayout layout;
	layout.model_name = model_name;

	layout.inputs = 
	{
		{ "x", "float32", { -1, 4, 1 } },
	};

	layout.outputs = 
	{
		{ "y" }
	};

	layout.layers = 
	{
		{ "Flatten",
			{
				{ "input_name", "x" },
				{ "output_name", "flat_input" }
			}
		},
		{ "Dense",
			{
				{ "input_name", "flat_input" },
				{ "units", 16 },
				{ "output_name", "dense_1" },
			}
		},
		{ "Dense", 
			{
				{ "input_name", "dense_1" },
				{ "units", 2 }, 
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
		std::cerr << "Failed to Execute Model Creation Python script. Exit code: " << exit_code << std::endl;
		return -1;
	}
	// --------------------------------------------------------------------------------------------


	// Create Training Data In C++ ----------------------------------------------------------------
	TF::TrainingBatch batch;

    // Input examples (2 samples of 4 features each)
    TF::NamedInput input;
    input.name = "x";
    input.data = 
	{
		{ 5.1, 3.5, 1.4, 0.2 }, // sample 1
		{ 6.2, 3.4, 5.4, 2.3 }, // sample 2
		{ 0.2, 6.8, 9.1, 1.2 }  // sample 3
    };

    // Label examples (2 samples, 3-class one-hot encoded)
    TF::NamedLabel label;
    label.name = "y";
    label.data = 
	{
		{ 1.0, 0.0 }, // class 0
		{ 0.0, 0.5 }, // class 2
		{ 0.0, 1.0 }  // class 2
    };

    // Add to batch
    batch.inputs.push_back(input);
    batch.labels.push_back(label);

	// Save to JSON
	batch.WriteToFile("train/train_data.json");
	// --------------------------------------------------------------------------------------------


	// Load The Model In C++ --------------------------------------------------------------
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
	cppflow::tensor input_x = cppflow::tensor(std::vector<float>{ 1.0f, 2.0f, 3.0f, 4.0f }, { 4, 1 });

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
	// --------------------------------------------------------------------------------------------


	// Run model
	std::vector<cppflow::tensor> results = model(inputs_vec, outputs_vec);

	// Output the pre-results ---------------------------------------------------------------------
	std::cout << "Input X:\n" << TF::PrintTensor<float>(input_x) << std::endl;
	
	std::cout << "Pre-Training Output" << std::endl;
	if (!results.empty()) 
	{
		for (uint32_t i = 0; i < results.size(); ++i)
		{
			std::cout << "Result {" << i << "}:" << std::endl;
			std::cout << "\t" << TF::PrintTensor<float>(results[i]) << std::endl;
		}
	}
	// --------------------------------------------------------------------------------------------


	// Train the Model In Python ------------------------------------------------------------------
	std::stringstream trainCmd;
	trainCmd << "python ../PythonScripts/train_model_from_json.py \"" << model_name << "\" \"model_description.json\" \"train/train_data.json\"";

	exit_code = std::system(trainCmd.str().c_str());
	if (exit_code != 0)
	{
		std::cerr << "Failed to Execute Training Python script. Exit code: " << exit_code << std::endl;
		return -1;
	}
	// --------------------------------------------------------------------------------------------


	// Run trained model
	cppflow::model trainedModel(model_name);
	results = trainedModel(inputs_vec, outputs_vec);

	// Output the trained results -----------------------------------------------------------------
	std::cout << "Post-Training Output" << std::endl;

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