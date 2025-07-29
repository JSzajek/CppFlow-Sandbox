#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace cppflow
{
	class tensor;
}

namespace TF
{
	class ImageTensorLoader 
	{
	public:
		enum class ChannelOrder 
		{
			GrayScale,

			BGR,
			BGRA,
			RGB,
			RGBA
		};

		enum class ShapeOrder
		{
			WidthHeightChannels,
			HeightWidthChannels,
			ChannelsHeightWidth,
			ChannelsWidthHeight
		};

		ImageTensorLoader(uint32_t width, 
						  uint32_t height, 
						  uint32_t channels,
						  bool normalize = true,
						  ChannelOrder order = ChannelOrder::RGBA,
						  ShapeOrder shape = ShapeOrder::WidthHeightChannels);

		bool Load(const std::string& image_path, cppflow::tensor& output);
	private:
		uint32_t mWidth = 0;
		uint32_t mHeight = 0;
		uint32_t mChannels = 0;
		bool mNormalize = true;
		ChannelOrder mChannelOrder = ChannelOrder::RGBA;
		ShapeOrder mShapeOrder = ShapeOrder::WidthHeightChannels;
	};
}