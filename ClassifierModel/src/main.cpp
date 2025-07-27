
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

#include <opencv2/opencv.hpp>

int main()
{
	const std::string model_name = "simple_mnist";
	const std::string model_description_path = model_name + "/model_description.json";

	// Create Model Description -------------------------------------------------------------------
	TF::ModelLayout layout;
	layout.model_name = model_name;

	layout.inputs = 
	{
		{ "input", "float32", { -1, 28, 28, 1 }, "image" },
	};

	layout.outputs = 
	{
		{ "class_probs" }
	};

	layout.layers = 
	{
		{ "Flatten",
			{
				{ "input_name", "input" },
				{ "output_name", "flat_input" }
			}
		},
		{ "Dense",
			{
				{ "input_name", "flat_input" },
				{ "units", 64 },
				{ "activation", "relu" },
				{ "output_name", "dense_output1" },
			}
		},
		{ "Dense",
			{
				{ "input_name", "dense_output1" },
				{ "units", 3 },
				{ "activation", "softmax" },
				{ "output_name", "class_probs" },
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

	// Load and pre-process the image
	cv::Mat image = cv::imread("digit.png", cv::IMREAD_GRAYSCALE);  // Load grayscale

	if (image.empty()) 
	{
		std::cerr << "Failed to load image\n";
		return -1;
	}

	// Resize to 28x28
	cv::resize(image, image, cv::Size(28, 28));

	// Normalize to [0, 1]
	image.convertTo(image, CV_32FC1, 1.0 / 255.0);

	// Flatten to 1D vector
	std::vector<float> input_data(image.begin<float>(), image.end<float>());

	// Create input tensor 
	cppflow::tensor input_tensor(input_data, { 1, 28, 28, 1 });


	// Match names from JSON to create input tuples
	if (io_names["inputs"].contains("input")) 
	{
		inputs_vec.emplace_back(io_names["inputs"]["input"].get<std::string>(), input_tensor);
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