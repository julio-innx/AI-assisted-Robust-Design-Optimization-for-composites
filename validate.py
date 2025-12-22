# validate.py
import numpy as np

data = np.load("composite_dataset.npz", allow_pickle=True)
print("Dataset keys:", list(data.keys()))
for k, v in data.items():
    if isinstance(v, np.ndarray):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {v}")
