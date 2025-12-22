# build_dataset.py
import torch
import numpy as np
import pandas as pd
import json
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import os
from utils import *

# parse_element_table('results/element_list.txt', 'element_data.txt')
# parse_dlist('results/bc_list.txt', 'bc_data.txt')
# parse_plist('results/pressure_list.txt', 'pressure_data.txt')

def load_config(config_file='config.json'):
    """Load section definitions from JSON"""
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    return {sec['id']: sec for sec in cfg['sections']}

def summarize_section(sec):
    """Convert ply stack into numerical features"""
    angles = [p['angle'] for p in sec['plies']]
    thicknesses = [p['thickness'] for p in sec['plies']]
    return [
        np.mean(angles),           # mean orientation
        np.std(angles),            # orientation spread
        np.sum(thicknesses),       # total thickness
        len(angles),               # number of plies
        angles[0],                 # top layer angle
        angles[-1]                 # bottom layer angle
    ]

def build_gnn_dataset(
    node_file='node_disp.txt',
    stress_file='stress_components.txt',
    elem_file='element_data.txt',
    config_file='config.json',
    output_file='composite_dataset.npz'
):
    print("🔍 Loading data...")

    # ================================
    # 1. Load Node Data
    # ================================
    try:
        df_node = pd.read_csv(node_file, sep=r'\s+', skiprows=1,
                              names=['NODE', 'X', 'Y', 'Z', 'DX', 'DY', 'DZ'], engine='python')
        df_stress = pd.read_csv(stress_file, sep=r'\s+', skiprows=1,
                                names=['NODE', 'SX', 'SY', 'SZ', 'SXY', 'SYZ', 'SXZ'], engine='python')

        # ✅ Ensure NODE is int
        df_node['NODE'] = pd.to_numeric(df_node['NODE'], errors='coerce').astype('int')
        df_stress['NODE'] = pd.to_numeric(df_stress['NODE'], errors='coerce').astype('int')

        # Drop rows with NaN
        df_node = df_node.dropna(subset=['NODE']).astype(int)
        df_stress = df_stress.dropna(subset=['NODE']).astype(int)

        # Merge
        df_node = df_node.merge(df_stress, on='NODE', how='inner')  # Only common nodes
        df_node = df_node.sort_values('NODE').reset_index(drop=True)

        # Map NODE ID to index
        node_id_to_idx = {row['NODE']: idx for idx, row in df_node.iterrows()}
        num_nodes = len(df_node)

        print(f"✅ Loaded {num_nodes} nodes")

    except Exception as e:
        print(f"❌ Error loading node/stress files: {e}")
        return None

    # ================================
    # 2. Load Element Data
    # ================================

    try:
        elem_data = parse_ansys_table(elem_file)
        if not elem_data:
            raise ValueError("No valid element data parsed")

        df_elem = pd.DataFrame(elem_data,
                               columns=['ELEM', 'MAT', 'TYP', 'REL', 'ESY', 'SEC', 'N1', 'N2', 'N3', 'N4'])

        # Convert to int
        node_cols = ['N1', 'N2', 'N3', 'N4']
        df_elem[node_cols] = df_elem[node_cols].apply(pd.to_numeric, errors='coerce')

        # Map node IDs to indices
        for col in node_cols:
            df_elem[f'{col}_idx'] = df_elem[col].map(node_id_to_idx).astype('Int64')

        # Keep only elements with valid nodes
        valid_mask = df_elem[[f'N{i}_idx' for i in range(1,5)]].notna().all(axis=1)
        df_elem = df_elem[valid_mask].copy()

        num_elems = len(df_elem)
        print(f"✅ Loaded {num_elems} elements")

    except Exception as e:
        print(f"❌ Error loading element file: {e}")
        return None

    # ================================
    # 3. Build Edge Index (from elements)
    # ================================
    edges = set()
    for _, elem in df_elem.iterrows():
        nodes = [elem[f'N{i}_idx'] for i in range(1,5)]
        for i in range(4):
            u, v = int(nodes[i]), int(nodes[(i+1)%4])
            edges.add((min(u, v), max(u, v)))

    if not edges:
        print("❌ No edges created. Check element connectivity.")
        return None

    edge_list = list(edges)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)

    print(f"✅ Built graph with {edge_index.size(1)//2} undirected edges")

    # ================================
    # 4. Load Section Features
    # ================================
    try:
        sections = load_config(config_file)
        section_features = {}
        for sec_id, sec in sections.items():
            section_features[int(sec_id)] = summarize_section(sec)

        # Assign to elements
        elem_sec_ids = torch.tensor(df_elem['SEC'].values, dtype=torch.long)
        elem_feature_list = [section_features[sec_id] for sec_id in df_elem['SEC']]
        elem_features = torch.tensor(elem_feature_list, dtype=torch.float)

        print("✅ Processed section features")

    except Exception as e:
        print(f"❌ Error processing sections: {e}")
        return None

    # Element to node mapping
    elem_node_cols = ['N1_idx', 'N2_idx', 'N3_idx', 'N4_idx']
    elem_to_node_np = df_elem[elem_node_cols].values.astype(np.int64)  # ← Force to int64
    elem_to_node = torch.tensor(elem_to_node_np, dtype=torch.long)      # ← Use long for indices

    # ================================
    # 5. Global Features (Load)
    # ================================
    try:
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        load_value = float(cfg['loads'][0]['value'])
        u_global = torch.tensor([[load_value]], dtype=torch.float)
        print(f"✅ Global input: Load = {load_value} MPa")
    except Exception as e:
        print(f"❌ Error reading load from config: {e}")
        # Global input
        u_global = torch.tensor([[load_value]], dtype=torch.float)


    # ================================
    # 6. Node Features (Output)
    # ================================
    pos = torch.tensor(df_node[['X', 'Y', 'Z']].values, dtype=torch.float)
    disp = torch.tensor(df_node[['DX', 'DY', 'DZ']].values, dtype=torch.float)
    stress = torch.tensor(df_node[['SX', 'SY', 'SXY']].values, dtype=torch.float)

    # Edge index (already fixed)
    edge_index = to_undirected(edge_index)

    # Element features
    elem_sec_ids = torch.tensor(df_elem['SEC'].values.astype(np.int64), dtype=torch.long)
    elem_feature_list = [section_features[sec_id] for sec_id in df_elem['SEC']]
    elem_features = torch.tensor(elem_feature_list, dtype=torch.float)

    # ================================
    # 7. Create PyG Data Object
    # ================================
    data = Data(
        pos=pos,
        disp=disp,
        stress=stress,
        edge_index=edge_index,
        elem_section_id=elem_sec_ids,
        elem_features=elem_features,
        elem_to_node=elem_to_node,
        u_global=u_global,
        num_nodes=num_nodes
    )

    # ================================
    # 8. Save Dataset
    # ================================
    np.savez(
        output_file,
        pos=data.pos.numpy(),
        disp=data.disp.numpy(),
        stress=data.stress.numpy(),
        edge_index=data.edge_index.numpy(),
        elem_section_id=data.elem_section_id.numpy(),
        elem_features=data.elem_features.numpy(),
        elem_to_node=data.elem_to_node.numpy(),
        u_global=data.u_global.numpy(),
        metadata={
            'num_nodes': num_nodes,
            'num_elements': len(elem_features),
            'feature_names': ['mean_angle', 'std_angle', 'total_thickness', 'n_plies', 'top_angle', 'bottom_angle']
        }
    )

    print(f"✅ Dataset saved to {output_file}")
    print(f"   Shape: {data}")
    return data

# ===========================
# Run the pipeline
# ===========================
# if __name__ == "__main__":
#     data = build_gnn_dataset()
#     if data is not None:
#         print("🎉 Dataset built successfully!")
#     else:
#         print("❌ Failed to build dataset.")
