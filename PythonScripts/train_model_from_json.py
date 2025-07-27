import json
import sys
import tensorflow as tf
import numpy as np

def tf_dtype_from_string(dtype_str):
    return {
        "float32": tf.float32,
        "float64": tf.float64,
        "int32": tf.int32,
        "int64": tf.int64,
        "uint8": tf.uint8,
        "bool": tf.bool
    }.get(dtype_str, tf.float32)

def load_model_description(desc_path):
    with open(desc_path, 'r') as f:
        return json.load(f)

def load_training_data(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)

def main(model_path, desc_path, data_path):
    # Load files
    layout = load_model_description(desc_path)
    train_data = load_training_data(data_path)

    # --- Load Keras model (not tf.saved_model.load!) ---
    model = tf.keras.models.load_model(model_path)

    # --- Prepare inputs and labels ---
    input_data = {}
    for input_spec in layout["inputs"]:
        name = input_spec["name"]
        dtype = tf_dtype_from_string(input_spec["dtype"])
        input_data[name] = tf.convert_to_tensor(train_data["inputs"][name], dtype=dtype)

    label_data = {}
    for output_spec in layout["outputs"]:
        name = output_spec["name"]
        label_data[name] = tf.convert_to_tensor(train_data["labels"][name], dtype=tf.float32)

    # --- Compile model ---
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # --- Fit model (assumes one input and one output for now) ---
    input_tensor = list(input_data.values())[0]
    label_tensor = list(label_data.values())[0]

    model.fit(input_tensor, label_tensor, epochs=10, batch_size=32)

    # --- Save updated model ---
    model.save("trained_" + model_path)
    print(f"Model retrained and saved to trained_{model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_model.py <model_path> <model_description.json> <train_data.json>")
        sys.exit(1)

    model_path = sys.argv[1]
    desc_path = sys.argv[2]
    data_path = sys.argv[3]
    main(model_path, desc_path, data_path)