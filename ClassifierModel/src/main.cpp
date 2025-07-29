
#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

#include <opencv2/opencv.hpp>

int main()
{
	TF::MLModel model("simple_mnist");

	int32_t target_width = 28;
	int32_t target_height = 28;

	model.AddInput( "input", 
				   TF::DataType::Float32,
				   { -1, target_width, target_height, 1 },
				   TF::DomainType::Image);

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



	TF::ImageTensorLoader image_loader(target_width, 
									   target_height, 
									   1, 
									   true, 
									   TF::ImageTensorLoader::ChannelOrder::GrayScale,
									   TF::ImageTensorLoader::ShapeOrder::WidthHeightChannels);


	std::unordered_map<std::string, cppflow::tensor> inputs;
	if (!image_loader.Load("digit.png", inputs["input"]))
	{
		std::cerr << "Failed to Load Image" << std::endl;
		return false;
	}

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