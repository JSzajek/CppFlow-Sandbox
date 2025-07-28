
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

#include <opencv2/opencv.hpp>

int main()
{
	TF::MLModel model("simple_mnist");

	model.AddInput("input", 
				   TF::DataType::Float32,
				   { -1, 28, 28, 1 }, 
				   TF::DomainType::Image);

	model.AddOutput("class_probs");

	model.AddLayer("Conv1D",
	{
		{ "input_name", "input" },
		{ "filters", 32 },
		{ "kernel_size", 3 },
		{ "output_name", "conv1d_1" },
	});

	model.AddLayer("Conv2D",
	{
		{ "input_name", "input" },
		{ "filters", 64 },
		{ "kernel_size", { 3, 3 } },
		{ "output_name", "conv2d_1" },
	});

	model.AddLayer("MaxPooling2D",
	{
		{ "input_name", "conv2d_1" },
		{ "pool_size", {2, 2} },
		{ "strides", {2, 2} },
		{ "padding", "valid"},
		{ "output_name", "pool1" },
	});

	model.AddLayer("Flatten",
	{
		{ "input_name", "pool1" },
		{ "output_name", "flat_input" }
	});

	model.AddLayer("Dense",
	{
		{ "input_name", "flat_input" },
		{ "units", 6 },
		{ "activation", "softmax" },
		{ "output_name", "class_probs" },
	});

	if (!model.CreateModel())
	{
		std::cerr << "Failed to create model." << std::endl;
		return -1;
	}


	// Example Input Image
	cv::Mat image = cv::imread("train/data/2.png", cv::IMREAD_GRAYSCALE);

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
	model.AddTrainingData("input", 
						  { "train/data/0.png" },
						  "class_probs", 
						  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f });

	model.AddTrainingData("input", 
						  { "train/data/1.png" },
						  "class_probs", 
						  { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f });

	model.AddTrainingData("input", 
						  { "train/data/2.png" },
						  "class_probs", 
						  { 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f });

	model.AddTrainingData("input", 
						  { "train/data/3.png" },
						  "class_probs", 
						  { 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f });

	model.AddTrainingData("input", 
						  { "train/data/4.png" },
						  "class_probs", 
						  { 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f });

	model.AddTrainingData("input", 
						  { "train/data/5.png" },
						  "class_probs", 
						  { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f });

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