#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include <nlohmann/json.hpp>

struct TFInput 
{
	// Desired input node name
	std::string name;

	// e.g. "float32"
	std::string dtype;

	// Shape of the input tensor, use -1 for dynamic dimensions
	std::vector<int> shape;
};

struct TFOutput 
{
	// Desired output node name
	std::string name; 
};

struct TFLayer 
{
	// e.g., "Dense", "Conv2D", "ReLU"
	std::string type; 

	// Generic parameters
	std::unordered_map<std::string, nlohmann::json> params; 
};

struct TFModelLayout 
{
	// Model name, e.g., "simple_add", "image_classifier"
	std::string model_name;

	// Model name, e.g., "simple_add", "image_classifier"
	std::vector<TFInput> inputs;

	// Input nodes, e.g., {"x", "float32", {-1}}, {"y", "float32", {-1}}
	std::vector<TFOutput> outputs;

	// Output nodes, e.g., {"add_result"}
	std::vector<TFLayer> layers;

	nlohmann::json to_json() const 
	{
		nlohmann::json j;
		j["model_name"] = model_name;

		for (const auto& input : inputs) 
		{
			j["inputs"].push_back(
			{
				{"name", input.name},
				{"dtype", input.dtype},
				{"shape", input.shape}
			});
		}

		for (const auto& output : outputs) 
		{
			j["outputs"].push_back(
			{
				{"name", output.name}
			});
		}

		for (const auto& layer : layers) 
		{
			j["layers"].push_back(
			{
				{"type", layer.type},
				{"params", layer.params}
			});
		}

		return j;
	}

	static TFModelLayout from_json(const nlohmann::json& j) 
	{
		TFModelLayout layout;
		layout.model_name = j.at("model_name");

		for (const auto& jin : j.at("inputs")) 
			layout.inputs.push_back({ jin.at("name"), jin.at("dtype"), jin.at("shape").get<std::vector<int>>() });

		for (const auto& jout : j.at("outputs")) 
			layout.outputs.push_back({ jout.at("name") });

		for (const auto& jlayer : j.at("layers")) 
			layout.layers.push_back({ jlayer.at("type"), jlayer.at("params") });

		return layout;
	}
};