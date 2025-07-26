# build_model_from_json.py
import tensorflow as tf
import json
import sys
from pathlib import Path
from tensorflow.python.tools import saved_model_utils
from tensorflow.core.protobuf import saved_model_pb2

def build_model(layout):
    inputs = {}
    for inp in layout["inputs"]:
        inputs[inp["name"]] = tf.keras.Input(shape=inp["shape"][1:], name=inp["name"], dtype=inp["dtype"])

    layers = dict(inputs)
    for layer in layout["layers"]:
        typ = layer["type"]
        params = layer["params"]
        if typ == "Add":
            layers[params["output_name"]] = tf.keras.layers.Add()([layers[n] for n in params["input_names"]])
        elif typ == "Dense":
            layers[params["output_name"]] = tf.keras.layers.Dense(
                units=params["units"],
                activation=params.get("activation", None)
            )(layers[params["input_name"]])
        # Extend with more layer types...

    outputs = [layers[o["name"]] for o in layout["outputs"]]
    model = tf.keras.Model(inputs=list(inputs.values()), outputs=outputs)
    return model

def extract_tensor_names(saved_model_dir, signature="serving_default"):
    meta_graph_def = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set="serve")
    sig_def = meta_graph_def.signature_def[signature]

    io_info = {
        "inputs": {},
        "outputs": {}
    }

    for k, v in sig_def.inputs.items():
        io_info["inputs"][k] = v.name  # This is what CppFlow wants

    for k, v in sig_def.outputs.items():
        io_info["outputs"][k] = v.name  # This is "PartitionedCall:0" etc.

    return io_info

def main(json_path):
    with open(json_path, "r") as f:
        layout = json.load(f)

    model_name = layout["model_name"]

    model = build_model(layout)
    tf.saved_model.save(model, model_name)

    # Extract Signatures
    io = extract_tensor_names(model_name)
    with open(model_name + "/cppflow_io_names.json", "w") as f:
        json.dump(io, f, indent=2)

if __name__ == "__main__":
    json_path = sys.argv[1]
    main(json_path)
