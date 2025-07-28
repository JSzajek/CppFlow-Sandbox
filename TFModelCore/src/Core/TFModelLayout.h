#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	enum class DataType
	{
		Bool,
		UInt8,
		Float32,
		Float64,
		Double,
		Int32,
		Int64,
	};

	enum class DomainType 
	{
		Data,
		Image,
	};

	struct Input
	{
	public:
		// Desired input node name
		std::string name;

		DataType type = DataType::Float32;

		// Shape of the input tensor, use -1 for dynamic dimensions
		std::vector<int> shape;

		DomainType domain = DomainType::Data;
	};

	struct Output
	{
	public:
		// Desired output node name
		std::string name;
	};

	struct Layer
	{
	public:
		// e.g., "Flatten", "Add", "Activation", Dense", "Dropout", "Conv2D", "MaxPooling2D", "BatchNormalization"
		std::string type;

		// Generic parameters
		std::unordered_map<std::string, nlohmann::json> params;
	};

	struct ModelLayout
	{
	public:
		void WriteToFile(const std::filesystem::path& filepath) const;
	private:
		nlohmann::json to_json() const;

		static ModelLayout from_json(const nlohmann::json& j);
	public:
		// Model name, e.g., "simple_add", "image_classifier"
		std::string model_name;

		// Input nodes, e.g., {"x", "float32", {-1}}, {"y", "float32", {-1}}
		std::vector<TF::Input> inputs;

		// Output nodes, e.g., {"add_result"}
		std::vector<TF::Output> outputs;

		// Layers in the model, e.g., {"Dense", {{"units", 64}, {"activation", "relu"}}}
		std::vector<TF::Layer> layers;
	};
}