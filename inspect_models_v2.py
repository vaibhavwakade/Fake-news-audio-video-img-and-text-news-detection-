import h5py
import torch
import pickle
import os
import tensorflow as tf
import numpy as np

def inspect_h5(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        model = tf.keras.models.load_model(file_path, compile=False)
        print(f"Model Name: {model.name}")
        model.summary()
        print(f"Input shape: {model.input_shape}")
        for i, layer in enumerate(model.layers):
            if i < 5 or i > len(model.layers) - 5:
                print(f"Layer {i}: {layer.name}, Type: {type(layer)}, Config: {layer.get_config().get('activation', 'N/A')}")
    except Exception as e:
        print(f"Error loading H5: {e}")

def inspect_pth(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        data = torch.load(file_path, map_location='cpu')
        print(f"Data type: {type(data)}")
        if isinstance(data, dict):
            print("Keys:", data.keys())
            # Try to see if it's a state_dict or a full model
            # Let's check some shapes
            for k in list(data.keys())[:10]:
                if hasattr(data[k], 'shape'):
                    print(f"  {k}: {data[k].shape}")
        else:
            print("Object summary:", data)
    except Exception as e:
        print(f"Error loading PTH: {e}")

def inspect_pkl(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Data type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"First 10 elements: {data[:10]}")
        elif hasattr(data, 'predict'):
            print("Detected a model with predict() method.")
            if hasattr(data, 'get_params'):
                print("Params:", data.get_params())
        else:
            print("Object content summary:", str(data)[:500])
    except Exception as e:
        print(f"Error loading PKL with pickle: {e}")
        try:
            import joblib
            data = joblib.load(file_path)
            print("Loaded with joblib. Data type:", type(data))
            if hasattr(data, 'predict'):
                print("Detected a model with predict() method.")
        except Exception as e2:
            print(f"Error loading PKL with joblib: {e2}")

if __name__ == "__main__":
    dataset_dir = "/Users/abhijeetgolhar/Downloads/P Project/Dataset"
    inspect_h5(os.path.join(dataset_dir, "image_train_model.h5"))
    inspect_h5(os.path.join(dataset_dir, "audio_train_model.h5"))
    inspect_pth(os.path.join(dataset_dir, "video_train_model.pth"))
    inspect_pkl(os.path.join(dataset_dir, "text_train_model.pkl"))
