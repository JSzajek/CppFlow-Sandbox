# **CppFlow Sandbox**
A lightweight C++/Python hybrid library for training, generating, converting, and deploying TensorFlow models using `cppflow`, OpenCV, and Python with `Tensorflow`.

## **Getting Started**

### **Prerequisites**
Ensure you have the following installed on your system:
- **C++20** or later
- A modern **C++ compiler** (e.g., GCC, Clang, MSVC)
- Python (Tested with 3.10.11)
  - Minimum Requirements
    - Tensorflow (2.13)
    - Keras (2.13.1)
    - Onnx2Keras (https://github.com/gmalivenko/onnx2keras)
    - Onnx (1.14.1)


### Supported Model Formats
- SavedModel (TensforFlow)
- ONNX

### **How to Build**
#### **Step 1: Clone the Repository**
```
git clone https://github.com/JSzajek/CppFlow-Sandbox.git
```

#### **Step 2: Setup Project Using Premake**
##### **Windows**
```
Run Win-GenerateProjects.bat
```

#### **Step 3: Build the Project**
##### **Windows**
Open the generated .sln file and build the project.

## **Features**
#### Model Definition and Training
- Build models using TensorFlow/Keras with a consistent, JSON-defined layout structure.
- Automatically supports conversion of `ONNX` to `SavedModel` format.
- Modular `ModelLayout` architecture allows easy customization of layers: Conv1D/2D, MaxPooling, Dense, Dropout, etc.
- Integrated training pipeline for classification tasks with automatic dataset loading and splitting.

#### Model Conversion Utilities
- Converts `ONNX` to TensorFlow `SavedModel`.
  - Conversion chain is ONNX to Keras to SavedModel.
- Extract and exports model meta data including input/output tensor names.
- Label map generation and export to JSON.


#### Image Pre-Processing & Tensor Conversion
- OpenCV based image loader that resized, normalizes, and converts images to any tensor layout.
- Flexible pixel access and image tensor packing based on user-defined shape order.
- SUpport automatic channel conversion.
- Seamless integration with `cppflow::tensor` for model inference.

## Example Usage
#### Model Creation 
```
TF::MLModel model("<model_name>");

model.AddInput("x", 
			   TF::DataType::Float32,
			   { -1 });

model.AddInput("y", 
			   TF::DataType::Float32,
			   { -1 });

model.AddOutput("add_result");

model.AddLayer(TF::LayerType::Add,
{
	{ "input_names", { "x", "y" } },
	{ "output_name", "add_result" }
});

if (!model.CreateModel())
{
	// Error Creating
}
```

#### Model Inference
```
TF::MLModel::Result results;
if (model.Run(inputs, results))
{
   // Print/Use Results
}
```

#### Image Pre-Processing
```
TF::ImageTensorLoader image_loader(target_width, 
                                   target_height, 
                                   1, 
                                   true, 
                                   TF::ChannelOrder::GrayScale,
                                   TF::ShapeOrder::WidthHeightChannels);


std::unordered_map<std::string, cppflow::tensor> inputs;
if (!image_loader.Load("<insert filepath>", inputs["<input label>"]))
{
	// Error Reading
}
```

## Samples
- Simple Add Model
- Linear Regression Model
- Image Classification Model
- Training Image Classification
- Training Linear Regression
- Loading Pre-Train Models


### Future Roadmap
- [ ] Embedded Python Environment
- [ ] Improved Training Workflow
- [ ] Runtime Efficient Inference API



## **License**
This project is licensed under the Apache License. See the LICENSE file for more details.