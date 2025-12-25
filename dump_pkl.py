
import pickle
import numpy as np

file_path = "/Users/abhijeetgolhar/Downloads/P Project/Dataset/text_train_model.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Content: {data}")
if isinstance(data, np.ndarray):
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
