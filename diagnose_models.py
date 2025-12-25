
import torch
import tensorflow as tf
import numpy as np
import os
import sys

# Add current dir to sys.path
sys.path.append(os.getcwd())
try:
    from detection_utils import Xception
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    from detection_utils import Xception

dataset_dir = os.path.join(os.getcwd(), "Dataset")

def test_video_model():
    print("\n--- Testing Video/Image Model (PyTorch) ---")
    model_path = os.path.join(dataset_dir, "video_train_model.pth")
    model = Xception(num_classes=2)
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("model.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Dummy image (all zeros)
    dummy_input = torch.zeros((1, 3, 299, 299))
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1).numpy()[0]
    print(f"Zero input probs: {probs}")
    
    # Dummy image (random)
    dummy_input = torch.randn((1, 3, 299, 299))
    with torch.no_grad():
        output = model(dummy_input)
        probs = torch.softmax(output, dim=1).numpy()[0]
    print(f"Random input probs: {probs}")

def test_audio_model():
    print("\n--- Testing Audio Model (Keras) ---")
    model_path = os.path.join(dataset_dir, "audio_train_model.h5")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Input shape is (None, 128, 109, 1)
    dummy_input = np.zeros((1, 128, 109, 1))
    probs = model.predict(dummy_input, verbose=0)[0]
    print(f"Zero input probs: {probs}")
    
    dummy_input = np.random.rand(1, 128, 109, 1)
    probs = model.predict(dummy_input, verbose=0)[0]
    print(f"Random input probs: {probs}")

if __name__ == "__main__":
    test_video_model()
    test_audio_model()
