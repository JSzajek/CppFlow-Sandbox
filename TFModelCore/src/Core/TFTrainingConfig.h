#pragma once

#include <string>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	struct TrainingConfig
	{
	public:
		void ReadFromFile(const std::filesystem::path& filepath);

		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		nlohmann::json to_json() const;

		static TrainingConfig from_json(const nlohmann::json& j);
	public:
		// Number of training epochs
		uint32_t epochs = 10;

		// Size of each training batch
		uint32_t batch_size = 32;

		// Learning rate for the optimizer
		float learning_rate = 0.001f;

		// Whether to shuffle the training data
		bool shuffle = true;

		// Fraction of data to reserve for validation
		float validation_split = 0.0f;
		

		//std::string optimizer = "adam";     // Optimizer to use (e.g., "adam", "sgd")
		//std::string loss_function = "mse";  // Loss function to use (e.g., "mse", "categorical_crossentropy")
		//std::vector<std::string> metrics;   // List of metrics to evaluate during training
		//std::string model_save_path;        // Path to save the trained model
		//std::string log_dir;                // Directory for logging training progress
		//bool early_stopping = false;        // Enable early stopping based on validation loss
		//float early_stopping_patience = 5;  // Number of epochs with no improvement before stopping
	};
}