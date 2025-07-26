#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	struct Input
	{
	public:
		// Desired input node name
		std::string name;

		// e.g. "float32"
		std::string dtype;

		// Shape of the input tensor, use -1 for dynamic dimensions
		std::vector<int> shape;
	};

	struct Outputs
	{
	public:
		// Desired output node name
		std::string name;
	};

	struct Layer
	{
	public:
		// e.g., "Dense", "Conv2D", "ReLU"
		std::string type;

		// Generic parameters
		std::unordered_map<std::string, nlohmann::json> params;
	};

	struct ModelLayout
	{
		// Model name, e.g., "simple_add", "image_classifier"
		std::string model_name;

		// Input nodes, e.g., {"x", "float32", {-1}}, {"y", "float32", {-1}}
		std::vector<TF::Input> inputs;

		// Output nodes, e.g., {"add_result"}
		std::vector<TF::Outputs> outputs;

		// Layers in the model, e.g., {"Dense", {{"units", 64}, {"activation", "relu"}}}
		std::vector<TF::Layer> layers;
	public:
		void WriteToFile(const std::filesystem::path& filepath);
	public:
		nlohmann::json to_json() const;
		
		static ModelLayout from_json(const nlohmann::json& j);
	};
}