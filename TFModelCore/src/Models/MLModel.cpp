#include "MLModel.h"

namespace TF
{
	MLModel::MLModel(const std::string& name)
		: mName(name)
	{
		mLayout.model_name = name;
	}

	MLModel::~MLModel()
	{
	}

	void MLModel::LoadFrom(const std::filesystem::path& loadpath)
	{
		std::string output_path = loadpath.string();
		mName = loadpath.stem().string();

		if (loadpath.has_extension() && loadpath.extension() == ".onnx")
		{
			output_path = (loadpath.parent_path() / loadpath.stem()).string();
			mName = output_path;

			if (!ConvertModelToSavedModel(loadpath, "./" + output_path))
				return;

		}

		if (!std::filesystem::exists(loadpath))
		{
			std::cerr << "Load Model Path Does Not Exist: " << loadpath << std::endl;
			return;
		}


		std::stringstream cmd;
		cmd << "python ../PythonScripts/extract_model_info.py"
			<< " \"" << output_path << "\"";

		int32_t exit_code = std::system(cmd.str().c_str());
		if (exit_code != 0)
		{
			std::cerr << "Failed to Extract Info From SavedModel {" << output_path << "} Exit code: " << exit_code << std::endl;
			return;
		}

		// Load JSON with input/output tensor names
		std::ifstream in(output_path + "/cppflow_io_names.json");
		if (!in.is_open())
		{
			std::cerr << "Failed to open cppflow_io_names.json" << std::endl;
			return;
		}

		nlohmann::json io_names;
		in >> io_names;

		for (auto& [key, val] : io_names["outputs"].items())
			mOutputIONames.push_back(val.get<std::string>());

		for (auto& [key, val] : io_names["inputs"].items())
			mInputToIONamesMap[key] = val.get<std::string>();

	}

	void MLModel::AddInput(const std::string& name,
						   DataType dtype,
						   std::vector<int> shape, 
						   DomainType domain)
	{
		mLayout.inputs.push_back(
		{ 
			name, 
			dtype, 
			shape, 
			domain 
		});
	}

	void MLModel::AddOutput(const std::string& name)
	{
		mLayout.outputs.push_back(
		{ 
			name 
		});
	}

	void MLModel::AddLayer(const std::string& type, 
						   const std::unordered_map<std::string, nlohmann::json>& params)
	{
		mLayout.layers.push_back(
		{ 
			type, 
			params 
		});
	}

	void MLModel::AddTrainingData(const std::string& input_name, 
								  const nlohmann::json& input_values,
								  const std::string& label_name,
								  const nlohmann::json& label_outputs)
	{
		NamedInput input;
		input.name = input_name;
		input.data = input_values;

		NamedLabel label;
		label.name = label_name; // Default label name, can be customized
		label.data = label_outputs;

		mCurrentTrainingBatch.inputs.push_back(input);
		mCurrentTrainingBatch.labels.push_back(label);
	}

	void MLModel::SaveLayoutJson(const std::filesystem::path& path) const
	{
		mLayout.WriteToFile(path);
	}

	void MLModel::SaveTrainingJson(const std::filesystem::path& path) const
	{
		mCurrentTrainingBatch.WriteToFile(path);
	}

	bool MLModel::CreateModel()
	{
		mOutputIONames.clear();

		// Write the layout to a file
		const std::string model_description_path = "./" + mName + "/model_description.json";
		mLayout.WriteToFile(model_description_path);


		// Run the Python script to create the model
		const std::string python_script = "python ../PythonScripts/build_model_from_json.py \"" + model_description_path + "\"";

		int32_t exit_code = std::system(python_script.c_str());
		if (exit_code != 0)
		{
			std::cerr << "Failed to Execute Model Creation Python script. Exit code: " << exit_code << std::endl;
			return false;
		}


		// Load JSON with input/output tensor names
		std::ifstream in(mName + "/cppflow_io_names.json");
		if (!in.is_open())
		{
			std::cerr << "Failed to open cppflow_io_names.json" << std::endl;
			return false;
		}

		nlohmann::json io_names;
		in >> io_names;

		for (auto& [key, val] : io_names["outputs"].items())
			mOutputIONames.push_back(val.get<std::string>());

		for (auto& [key, val] : io_names["inputs"].items())
			mInputToIONamesMap[key] = val.get<std::string>();

		return true;
	}

	bool MLModel::TrainModel(const std::string& output_name, 
							 uint32_t epochs,
							 uint32_t batchSize, 
							 float learning_rate, 
							 bool shuffle, 
							 float validation_split)
	{
		if (mCurrentTrainingBatch.inputs.empty() || mCurrentTrainingBatch.labels.empty())
			return false;


		TF::TrainingConfig config;
		config.epochs = epochs;
		config.batch_size = batchSize;
		config.learning_rate = learning_rate;
		config.shuffle = shuffle;
		config.validation_split = validation_split;

		config.WriteToFile("train/train_config.json");


		mCurrentTrainingBatch.WriteToFile("train/train_data.json");

		std::string outputName = output_name.empty() ? mName : output_name;

		std::stringstream trainCmd;
		trainCmd << "python ../PythonScripts/train_model_from_json.py"
				 << " \"" << mName << "\""
				 << " \"" << outputName << "\""
				 << " \"train/train_config.json\""
				 << " \"train/train_data.json\"";

		int32_t exit_code = std::system(trainCmd.str().c_str());
		if (exit_code != 0)
		{
			std::cerr << "Failed to Execute Training Python script. Exit code: " << exit_code << std::endl;
			return false;
		}
		return true;
	}

	bool MLModel::Run(const std::unordered_map<std::string, cppflow::tensor>& input_tensors,
					  Result& output)
	{
		if (mOutputIONames.empty())
			return false;

		std::vector<std::tuple<std::string, cppflow::tensor>> inputs_vec;
		for (const auto& [name, tensor] : input_tensors)
		{
			auto found = mInputToIONamesMap.find(name);
			if (found == mInputToIONamesMap.end())
			{
				std::cerr << "Input name '" << name << "' not found in model input names." << std::endl;
				continue;
			}

			inputs_vec.emplace_back(found->second, tensor);
		}

		cppflow::model model(mName);
		std::vector<cppflow::tensor> results = model(inputs_vec, mOutputIONames);

		for (size_t i = 0; i < results.size(); ++i)
		{
			const auto& output_name = mOutputIONames[i];
			output[output_name] = results[i];
		}
		return true;
	}

	void MLModel::ExportAll(const std::filesystem::path& directory) const
	{
		std::filesystem::path dir_path(directory);
		if (!std::filesystem::exists(dir_path))
			std::filesystem::create_directories(dir_path);

		mLayout.WriteToFile(dir_path / "model_layout.json");
		mCurrentTrainingBatch.WriteToFile(dir_path / "train_data.json");

		std::cout << "Model and Training Data Exported to: " << dir_path << std::endl;
	}

	bool MLModel::ConvertModelToSavedModel(const std::filesystem::path& filepath,
										   const std::filesystem::path& outputpath)
	{
		std::stringstream cmd;
		cmd << "python ../PythonScripts/convert_onnx_to_saved_model.py"
				 << " \"" << filepath.string() << "\""
				 << " \"" << outputpath.string() << "\"";

		int32_t exit_code = std::system(cmd.str().c_str());
		if (exit_code != 0)
		{
			std::cerr << "Failed to convert model to SavedModel format. Exit code: " << exit_code << std::endl;
			return false;
		}
		return true;	
	}
}