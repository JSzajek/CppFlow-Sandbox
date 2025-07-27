#include "TFModelLayout.h"

#include <iostream>
#include <fstream>

namespace TF
{
	void ModelLayout::WriteToFile(const std::filesystem::path& filepath)
	{
		nlohmann::json j = to_json();
		std::ofstream("model_description.json") << j.dump(2);
	}


	nlohmann::json ModelLayout::to_json() const
	{
		nlohmann::json result;
		result["model_name"] = model_name;

		for (const auto& input : inputs)
		{
			result["inputs"].push_back(
			{
				{ "name", input.name },
				{ "dtype", input.dtype },
				{ "shape", input.shape },
				{ "input_type", input.input_type }
			});
		}

		for (const auto& output : outputs)
		{
			result["outputs"].push_back(
			{
				{ "name", output.name }
			});
		}

		for (const auto& layer : layers)
		{
			result["layers"].push_back(
			{
				{ "type", layer.type },
				{ "params", layer.params }
			});
		}

		return result;
	}

	ModelLayout ModelLayout::from_json(const nlohmann::json& j)
	{
		ModelLayout layout;
		layout.model_name = j.at("model_name");

		for (const auto& jin : j.at("inputs"))
		{
			layout.inputs.push_back(
			{ 
				jin.at("name"), 
				jin.at("dtype"), 
				jin.at("shape").get<std::vector<int>>(),
				jin.value("input_type", "data") // Default to "data" if not specified
			});

		}

		for (const auto& jout : j.at("outputs"))
		{
			layout.outputs.push_back(
			{ 
				jout.at("name") 
			});
		}

		for (const auto& jlayer : j.at("layers"))
		{
			layout.layers.push_back(
			{ 
				jlayer.at("type"), 
				jlayer.at("params") 
			});
		}

		return layout;
	}
}