#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>

#include <nlohmann/json.hpp>

namespace TF
{
	struct NamedInput
	{
		std::string name;                   // Must match ModelLayout.input.name
		std::vector<nlohmann::json> data;   // Outer vector = batch, inner = flat feature array
	};

	struct NamedLabel
	{
		std::string name;                   // Must match ModelLayout.output.name
		std::vector<nlohmann::json> data;   // e.g., one-hot vectors
	};

	struct TrainingBatch
	{
	public:
		void ReadFromFile(const std::filesystem::path& filepath);

		void WriteToFile(const std::filesystem::path& filepath);
	private:
		nlohmann::json to_json() const;

		static TrainingBatch from_json(const nlohmann::json& j);
	public:
		std::vector<NamedInput> inputs;

		std::vector<NamedLabel> labels;
	};
}