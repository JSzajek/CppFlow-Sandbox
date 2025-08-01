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
	/// <summary>
	/// Class representing a Machine Learning Model that can be used for training and inference.
	/// </summary>
	class MLModel
	{
	public:
		using Result = std::unordered_map<std::string, cppflow::tensor>;
	public:
		/// <summary>
		/// Constructor initializing a MLModel.
		/// </summary>
		/// <param name="modelpath">The model path</param>
		MLModel(const std::filesystem::path& modelpath);
	public:
		/// <summary>
		/// Loads Pre-Trained Models. Currently Only Supports loading ONNX or SavedModel format models.
		/// 
		/// Non-SavedModel formats will be converted to SavedModel format.
		/// </summary>
		/// <param name="loadpath">The filepath to load from</param>
		void LoadFrom(const std::filesystem::path& loadpath);

		/// <summary>
		/// Adds an input to the model.
		/// </summary>
		/// <param name="name">The name of the input</param>
		/// <param name="dtype">The data type of the input</param>
		/// <param name="shape">The shape of the input</param>
		/// <param name="domain">The domain type of the input</param>
		void AddInput(const std::string& name,
					  DataType dtype,
					  std::vector<int> shape, 
					  DomainType domain = DomainType::Data);

		/// <summary>
		/// Adds an output to the model.
		/// </summary>
		/// <param name="name">The output label</param>
		void AddOutput(const std::string& name);

		/// <summary>
		/// Adds a layer to the model.
		/// </summary>
		/// <param name="type">The type of the layer</param>
		/// <param name="params">The parameters of the layer</param>
		void AddLayer(const std::string& type, 
					  const std::unordered_map<std::string, nlohmann::json>& params);

		/// <summary>
		/// Adds training data to the model.
		/// </summary>
		/// <param name="input_name">The input name of the training batch</param>
		/// <param name="input_values">The input values</param>
		/// <param name="label_name">The label name of the training batch</param>
		/// <param name="label_outputs">The label outputs</param>
		void AddTrainingData(const std::string& input_name, 
						     const nlohmann::json& input_values,
						     const std::string& label_name,
						     const nlohmann::json& label_outputs);

		/// <summary>
		/// Save the model layout to a JSON file.
		/// </summary>
		/// <param name="path">The output path of the json file</param>
		void SaveLayoutJson(const std::filesystem::path& path) const;

		/// <summary>
		/// Save the training data to a JSON file.
		/// </summary>
		/// <param name="path">The output path of the json file</param>
		void SaveTrainingJson(const std::filesystem::path& path) const;

		/// <summary>
		/// Creates the model based on the current layout and training data.
		/// </summary>
		/// <returns>True if the creation was successful</returns>
		bool CreateModel();

		/// <summary>
		/// Launches the training of the model.
		/// </summary>
		/// <param name="output_path">The output path of the model</param>
		/// <param name="epochs">The number of epochs</param>
		/// <param name="batchSize">The batch size</param>
		/// <param name="learning_rate">The learning rate</param>
		/// <param name="shuffle">Whether to shuffle the training</param>
		/// <param name="validation_split">The validation split</param>
		/// <returns>True if the training was successful</returns>
		bool TrainModel(const std::string& output_path,
						uint32_t epochs = 10,
						uint32_t batchSize = 32,
						float learning_rate = 0.001f,
						bool shuffle = true,
						float validation_split = 0.0f);

		/// <summary>
		/// Launches the training of the model.
		/// </summary>
		/// <param name="epochs">The number of epochs</param>
		/// <param name="batchSize">The batch size</param>
		/// <param name="learning_rate">The learning rate</param>
		/// <param name="shuffle">Whether to shuffle the training</param>
		/// <param name="validation_split">The validation split</param>
		/// <returns>True if the training was successful</returns>
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

		/// <summary>
		/// Runs the model with the given input tensors and returns the output.
		/// </summary>
		/// <param name="input_tensors">The input tensors</param>
		/// <param name="output">The output result</param>
		/// <returns>True if the running the model was successful</returns>
		bool Run(const std::unordered_map<std::string, cppflow::tensor>& input_tensors,
				 Result& output);

		/// <summary>
		/// Exports all of the model's components to the specified directory.
		/// </summary>
		/// <param name="directory">The output directory</param>
		void ExportAll(const std::filesystem::path& directory) const;
	private:
		/// <summary>
		/// Converts the model to a SavedModel format if it is not already in that format.
		/// </summary>
		/// <param name="filepath">The input model's filepath</param>
		/// <param name="outputpath">The output model path</param>
		/// <returns>True if conversion was successful</returns>
		bool ConvertModelToSavedModel(const std::filesystem::path& filepath,
									  const std::filesystem::path& outputpath);
	public:
		std::string mModelPath;

		ModelLayout mLayout;

		std::unordered_map<std::string, std::string> mInputToIONamesMap;
		std::vector<std::string> mOutputIONames;

		TrainingBatch mCurrentTrainingBatch;
	};
}