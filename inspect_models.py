import h5py
import torch
import pickle
import os
import tensorflow as tf

def inspect_h5(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        model = tf.keras.models.load_model(file_path, compile=False)
        model.summary()
        print(f"Input shape: {model.input_shape}")
    except Exception as e:
        print(f"Error loading H5: {e}")

def inspect_pth(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        data = torch.load(file_path, map_location='cpu')
        if isinstance(data, dict):
            print("Keys in state_dict:", data.keys())
        else:
            print("Type of loaded object:", type(data))
            if hasattr(data, 'eval'):
                print("Architecture summary (first layer):", list(data.modules())[0])
    except Exception as e:
        print(f"Error loading PTH: {e}")

def inspect_pkl(file_path):
    print(f"\n--- Inspecting {file_path} ---")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print("Type of loaded object:", type(data))
        if hasattr(data, 'get_params'):
            print("Model parameters:", data.get_params())
    except Exception as e:
        print(f"Error loading PKL: {e}")

if __name__ == "__main__":
    dataset_dir = "/Users/abhijeetgolhar/Downloads/P Project/Dataset"
    inspect_h5(os.path.join(dataset_dir, "image_train_model.h5"))
    inspect_h5(os.path.join(dataset_dir, "audio_train_model.h5"))
    inspect_pth(os.path.join(dataset_dir, "video_train_model.pth"))
    inspect_pkl(os.path.join(dataset_dir, "text_train_model.pkl"))
