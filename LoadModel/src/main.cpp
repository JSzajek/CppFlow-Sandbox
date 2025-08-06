#include <iostream>
#include <cstdlib>

#include "TFModelLib.h"

#include <opencv2/opencv.hpp>

int main()
{
	// Requires Running The DownloadBirdDataset.py Script To Download and Extract The Dataset

	TF::MLModel model("BirdClassifier");
	model.LoadFrom("./model/bird-classifier/BirdClassifier.onnx");

	int32_t target_width = 260;
	int32_t target_height = 260;

	TF::ImageTensorLoader image_loader(target_width, 
									   target_height, 
									   3, 
									   true, 
									   TF::ChannelOrder::RGB,
									   TF::ShapeOrder::ChannelsHeightWidth);
	
	// Load Labels
	std::ifstream in("data/label_map.json");
	if (!in.is_open())
	{
		std::cerr << "Failed to open label_map.json" << std::endl;
		return false;
	}

	nlohmann::json label_map;
	in >> label_map;



	std::unordered_map<std::string, cppflow::tensor> inputs;

	const auto TestImage = [&](const std::string& imagepath) -> bool
	{
		if (!image_loader.Load(imagepath, inputs["pixel_values"]))
		{
			std::cerr << "Failed to Load Image" << std::endl;
			return false;
		}

		TF::MLModel::LabeledTensor results;
		if (model.Run(inputs, results))
		{
			// Output the results -------------------------------------------------------------------------

			std::string category_name = "UNKNOWN";
			if (!results.empty())
			{
				for (const auto& [key, value] : results)
				{
					const std::vector<float> pred = value.get_data<float>();
					uint32_t max_index = static_cast<uint32_t>(std::distance(pred.begin(), std::max_element(pred.begin(), pred.end())));

					category_name = label_map.value(std::to_string(max_index), "UNKNOWN");
					std::cout << "Predicted Class Index: " << max_index << " Name: " << category_name << std::endl;
				}
			}

			cv::Mat image = cv::imread(imagepath, cv::IMREAD_UNCHANGED);
			cv::resize(image, image, cv::Size(512, 512));

			cv::imshow(category_name, image);
			cv::waitKey();

			// --------------------------------------------------------------------------------------------
		}
		else
		{
			std::cerr << "Failed to Run Model." << std::endl;
			return false;
		}
		return true;
	};


	// Test Images ------------------------------------------------------------
	TestImage("data/test_bird_dataset/31/ANNAS HUMMINGBIRD.jpg"); // Correct
	TestImage("data/test_bird_dataset/158/COMMON HOUSE MARTIN.jpg"); // Incorrect
	TestImage("data/test_bird_dataset/306/IVORY GULL.jpg"); // Correct
	// ------------------------------------------------------------------------
	return 0;
}