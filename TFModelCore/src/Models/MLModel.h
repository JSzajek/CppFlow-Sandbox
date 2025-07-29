#pragma once

#include "Core/TFUtilities.h"
#include "Core/TFModelLayout.h"
#include "Core/TFTrainingBatch.h"
#include "Core/TFTrainingConfig.h"

#include <vector>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace TF
{
	class MLModel
	{
	public:
		using Result = std::unordered_map<std::string, cppflow::tensor>;
	public:
		MLModel(const std::string& name);

		~MLModel();
	public:
		/// <summary>
		/// Loads Pre-Trained Models. Currently Only Supports loading ONNX or SavedModel format models.
		/// 
		/// Non-SavedModel formats will be converted to SavedModel format.
		/// </summary>
		/// <param name="loadpath"></param>
		void LoadFrom(const std::filesystem::path& loadpath);

		void AddInput(const std::string& name,
					  DataType dtype,
					  std::vector<int> shape, 
					  DomainType domain = DomainType::Data);

		void AddOutput(const std::string& name);

		void AddLayer(const std::string& type, 
					  const std::unordered_map<std::string, nlohmann::json>& params);


		void AddTrainingData(const std::string& input_name, 
						     const nlohmann::json& input_values,
						     const std::string& label_name,
						     const nlohmann::json& label_outputs);

		// Save model + data JSON files
		void SaveLayoutJson(const std::filesystem::path& path) const;
		void SaveTrainingJson(const std::filesystem::path& path) const;

		bool CreateModel();

		// Launch Python training
		bool TrainModel(const std::string& output_name,
						uint32_t epochs = 10,
						uint32_t batchSize = 32,
						float learning_rate = 0.001f,
						bool shuffle = true,
						float validation_split = 0.0f);

		bool TrainModel(uint32_t epochs = 10,
						uint32_t batchSize = 32,
						float learning_rate = 0.001f,
						bool shuffle = true,
						float validation_split = 0.0f)
		{
			return TrainModel("", 
							  epochs, 
							  batchSize, 
							  learning_rate, 
							  shuffle, 
							  validation_split);
		}

		// Inference
		bool Run(const std::unordered_map<std::string, cppflow::tensor>& input_tensors,
				 Result& output);

		// Utility
		void ExportAll(const std::filesystem::path& directory) const;
	private:
		bool ConvertModelToSavedModel(const std::filesystem::path& filepath,
									  const std::filesystem::path& outputpath);
	public:
		std::string mName;

		ModelLayout mLayout;

		std::unordered_map<std::string, std::string> mInputToIONamesMap;
		std::vector<std::string> mOutputIONames;

		TrainingBatch mCurrentTrainingBatch;
	};
}