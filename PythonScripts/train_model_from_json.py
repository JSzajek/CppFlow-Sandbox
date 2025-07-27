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

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def main(model_path, desc_path, data_path):
    # Load files --------------------------------------------------------------
    layout = load_json(desc_path)
    train_data = load_json(data_path)

    # --- Load Keras model ----------------------------------------------------
    model = tf.keras.models.load_model(model_path)
    # -------------------------------------------------------------------------


    # --- Prepare inputs and labels -------------------------------------------
    input_data = {}
    for input_spec in layout["inputs"]:
        name = input_spec["name"]
        dtype = tf_dtype_from_string(input_spec["dtype"])
        shape = input_spec["shape"]
        intype = input_spec["input_type"]

        tensor = tf.convert_to_tensor(train_data["inputs"][name], dtype=dtype)

        # Determine expected shape (ignore -1 for batch size)
        target_shape = [dim for dim in shape if dim != -1]
        if target_shape:
            tensor = tf.reshape(tensor, [-1] + target_shape)

        input_data[name] = tensor

    label_data = {}
    for output_spec in layout["outputs"]:
        name = output_spec["name"]
        raw_label = train_data["labels"][name]
        tensor = tf.convert_to_tensor(raw_label, dtype=tf.float32)
        label_data[name] = tensor
    # -------------------------------------------------------------------------


    # --- Compile model -------------------------------------------------------
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # -------------------------------------------------------------------------

    # --- Fit model -----------------------------------------------------------
    model.fit(x=input_data, y=label_data, epochs=10, batch_size=32)
    # -------------------------------------------------------------------------

    # --- Save updated model --------------------------------------------------
    model.save(model_path)
    print(f"Model retrained and saved to {model_path}")
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python train_model.py <model_path> <output_path> <model_description.json> <train_data.json>")
        sys.exit(1)

    model_path = sys.argv[1]
    output_model_path = sys.argv[2]
    model_path = sys.argv[1]
    desc_path = sys.argv[2]
    data_path = sys.argv[3]
    main(model_path, desc_path, data_path)