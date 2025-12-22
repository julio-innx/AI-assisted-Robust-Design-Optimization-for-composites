# gnn/predict.py
import torch
import numpy as np
from pathlib import Path
from gnn.load_data import load_composite_dataset
from gnn.model import CompositeGNN

# Global variables
model = None
scalers = None
ref_graph = None


def load_model_and_scalers(dataset_file="composite_dataset.npz"):
    """
    Load trained model, scalers, and reference graph.
    Architecture parameters are loaded from the checkpoint.
    Returns True if successful.
    """
    global model, scalers, ref_graph

    try:
        # Check required files
        required_files = ["composite_gnn.pth", "scalers.npz", dataset_file]
        missing = [f for f in required_files if not Path(f).exists()]
        if missing:
            print(f"❌ Missing files: {missing}")
            return False

        # Load scalers
        data_npz = np.load("scalers.npz", allow_pickle=True)
        required_keys = ['disp_mean', 'disp_std', 'coord_mean', 'coord_std']
        missing = [k for k in required_keys if k not in data_npz]
        if missing:
            print(f"❌ Missing in scalers.npz: {missing}")
            return False

        scalers = {
            'disp': type('Scaler', (), {'mean_': data_npz['disp_mean'], 'scale_': data_npz['disp_std']}),
            'coord': type('Scaler', (), {'mean_': data_npz['coord_mean'], 'scale_': data_npz['coord_std']})
        }

        # Load dataset to get ref_graph
        try:
            dataset, _ = load_composite_dataset(dataset_file)
            ref_graph = dataset[0]
            print(f"✅ Loaded reference graph with {ref_graph.num_nodes} nodes")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False

        # ✅ Load checkpoint and extract architecture
        ckpt = torch.load("composite_gnn.pth", map_location='cpu')
        
        # Handle both old and new checkpoint formats
        if isinstance(ckpt, dict):
            if 'in_channels' in ckpt and 'hidden_channels' in ckpt and 'out_channels' in ckpt:
                # New format with architecture info
                in_channels = ckpt['in_channels']
                hidden_channels = ckpt['hidden_channels']
                out_channels = ckpt['out_channels']
                state_dict = ckpt['model']
                print(f"✅ Loaded architecture: in={in_channels}, hidden={hidden_channels}, out={out_channels}")
            elif 'model' in ckpt:
                # Old format - try to infer from ref_graph
                print("⚠️  Old checkpoint format detected. Inferring architecture...")
                in_channels = ref_graph.x.shape[1]
                hidden_channels = 64  # Default fallback
                out_channels = 9
                state_dict = ckpt['model']
            else:
                # Very old format - direct state dict
                print("⚠️  Very old checkpoint format. Using default architecture...")
                in_channels = ref_graph.x.shape[1]
                hidden_channels = 64
                out_channels = 9
                state_dict = ckpt
        else:
            # Direct state dict (oldest format)
            print("⚠️  Legacy checkpoint format. Using default architecture...")
            in_channels = ref_graph.x.shape[1]
            hidden_channels = 64
            out_channels = 9
            state_dict = ckpt

        # Create model with loaded architecture
        model = CompositeGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels
        )
        
        model.load_state_dict(state_dict)
        model.eval()

        print("✅ Model, scalers, and ref_graph loaded successfully!")
        print(f"   Architecture: {in_channels} → {hidden_channels} → {out_channels}")
        return True

    except Exception as e:
        print(f"❌ Failed to load model/scalers: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_laminate(angles, thicknesses, material_ids=None):
    """
    Predict displacement field only (for now).
    """
    global model, scalers, ref_graph

    # 🔍 Force reload if any component is missing
    if model is None or scalers is None or ref_graph is None:
        print("⚠️  One or more components not loaded. Forcing reload...")
        success = load_model_and_scalers()
        if not success:
            raise RuntimeError("❌ Failed to load model, scalers, or ref_graph after retry.")

    if len(angles) != len(thicknesses):
        raise ValueError("Angles and thicknesses must have same length")

    # ✅ Now safe to use ref_graph
    try:
        max_plies = (ref_graph.x.shape[1] - 3) // 3
        N_nodes = ref_graph.num_nodes
    except AttributeError as e:
        raise RuntimeError(f"ref_graph is invalid: {e}")

    if material_ids is None:
        material_ids = [1] * len(angles)
    if len(material_ids) != len(angles):
        raise ValueError("material_ids must match number of plies")

    def pad(seq, val=0.0):
        return (seq + [val] * max_plies)[:max_plies]

    angles_padded = pad(angles)
    thick_padded = pad(thicknesses)
    matid_padded = pad(material_ids, val=0.0)

    u_flat = []
    for i in range(max_plies):
        u_flat += [angles_padded[i], thick_padded[i], matid_padded[i]]

    u = torch.tensor(u_flat, dtype=torch.float32).unsqueeze(0).expand(N_nodes, -1)
    pos_scaled = ref_graph.pos
    x = torch.cat([pos_scaled, u], dim=1)

    data = type('Data', (), {})()
    data.x = x
    data.edge_index = ref_graph.edge_index
    data.batch = None

    with torch.no_grad():
        pred_scaled = model(data)

    disp_scaled = pred_scaled[:, :3].numpy()
    disp_pred = disp_scaled * scalers['disp'].scale_ + scalers['disp'].mean_
    coords_original = ref_graph.pos.numpy() * scalers['coord'].scale_ + scalers['coord'].mean_

    return disp_pred, coords_original
