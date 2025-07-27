
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

#include <opencv2/opencv.hpp>

int main()
{
	TF::MLModel model("simple_mnist");

	model.AddInput("input", "float32", { -1, 28, 28, 1 }, "image");

	model.AddOutput("class_probs");

	model.AddLayer("Flatten",
	{
		{ "input_name", "input" },
		{ "output_name", "flat_input" }
	});

	model.AddLayer("Dense",
	{
		{ "input_name", "flat_input" },
		{ "units", 64 },
		{ "activation", "relu" },
		{ "output_name", "dense_output1" },
	});

	model.AddLayer("Dense",
	{
		{ "input_name", "dense_output1" },
		{ "units", 3 },
		{ "activation", "softmax" },
		{ "output_name", "class_probs" },
	});

	if (!model.CreateModel())
	{
		std::cerr << "Failed to create model." << std::endl;
		return -1;
	}


	// Example Input Image
	cv::Mat image = cv::imread("digit.png", cv::IMREAD_GRAYSCALE);

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

	std::unordered_map<std::string, cppflow::tensor> inputs;
	inputs["input"] = input_tensor;

	TF::MLModel::Result results;
	if (model.Run(inputs, results))
	{
		// Output the results -------------------------------------------------------------------------
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