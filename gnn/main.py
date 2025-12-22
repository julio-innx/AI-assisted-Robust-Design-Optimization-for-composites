from load_data import load_composite_dataset
from train import train_gnn
import torch
from torch_geometric.data import Data

def validate_graph(data, idx):
    if data.x is not None and (torch.isnan(data.x).any() or torch.isinf(data.x).any()):
        print(f"❌ Graph {idx}: NaN/Inf in x")
        return False
    if torch.isnan(data.y).any() or torch.isinf(data.y).any():
        print(f"❌ Graph {idx}: NaN/Inf in y")
        return False
    if data.edge_index.max() >= data.num_nodes:
        print(f"❌ Graph {idx}: edge_index out of bounds")
        return False
    if data.y.shape[0] != data.num_nodes:
        print(f"❌ Graph {idx}: y rows != nodes")
        return False
    return True

# --- Load & Validate ---
print("🔍 Loading dataset...")
dataset, scalers = load_composite_dataset("composite_dataset.npz")

print("🧹 Filtering invalid graphs...")
cleaned_dataset = [g for i, g in enumerate(dataset) if validate_graph(g, i)]

if not cleaned_dataset:
    raise ValueError("❌ All graphs are invalid! Please check ANSYS results for errors.")

print(f"✅ Cleaned dataset: {len(cleaned_dataset)} valid simulations")

# --- Train ---
model = train_gnn(cleaned_dataset, epochs=200, lr=3e-5, hidden=64)
