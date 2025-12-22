import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

def parse_materials_from_configs(configs_dir="configs"):
    """
    Parse all config_XXXX.json files to extract materials and plies from the first section only.
    """
    configs_dir = Path(configs_dir)
    config_files = sorted(configs_dir.glob("config_*.json"))
    materials_list = []

    for cfg_file in config_files:
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)

        # Parse materials into dict by ID
        mat_dict_by_id = {}
        for mat in cfg['materials']:
            mat_id = int(mat['id'])
            mat_dict_by_id[mat_id] = {
                'EX': float(mat['EX']),
                'EY': float(mat['EY']),
                'GXY': float(mat['GXY']),
                'PRXY': float(mat['PRXY']),
                'EZ': float(mat.get('EZ', mat['EY'])),
                'GYZ': float(mat.get('GYZ', mat['GXY'])),
                'GXZ': float(mat.get('GXZ', mat['GXY'])),
                'PRYZ': float(mat.get('PRYZ', mat['PRXY'])),
                'PRXZ': float(mat.get('PRXZ', mat['PRXY']))
            }

        # Use only the first section's plies (to match .npz construction)
        sections = cfg.get('sections', [])
        plies = []
        if sections:
            for p in sections[0]['plies']:
                plies.append({
                    'angle': float(p['angle']),
                    'thickness': float(p['thickness']),
                    'material_id': int(p['material_id'])
                })

        materials_list.append({
            'materials': mat_dict_by_id,
            'plies': plies  # Flat list of dicts from first section
        })
    return materials_list

#def load_composite_dataset(dataset_file='composite_dataset.npz'):
#    data = np.load(dataset_file, allow_pickle=True)
#    
#    # Extract arrays
#    x_node = data['x']           # [N_cases, N_nodes, 3]
#    y_disp = data['y_disp']      # [N_cases, N_nodes, 3]
#    #y_stress = data['y_stress_comp']  # [N_cases, N_nodes, 6] ← Must exist!
#    layup_angles = data['layup_angles']
#    layup_thicknesses = data['layup_thicknesses']
#    edge_index_np = data['edge_index']
##    plies_list = data['plies'].tolist()
##    materials_dict = data['materials'].item()
#    plies_padded = data['plies']          # [N_cases, max_plies, 3]
#    plies_mask   = data['plies_mask']     # bool mask
#    #materials_dict = data['materials'][0] if isinstance(data['materials'], np.ndarray) else data['materials']
#    mat = data['materials']
#    if isinstance(mat, np.ndarray):
#        if mat.ndim == 0:
#            materials_dict = mat[()]           # 0-dim scalar array → use [()]
#        elif mat.size == 1:
#            materials_dict = mat.item()        # or mat[0] works too
#        else:
#            materials_dict = mat               # shouldn't happen
#    else:
#        materials_dict = mat
#
#    N_cases, N_nodes, _ = x_node.shape
#
#    # -------------------------------
#    # Normalize ALL outputs
#    # -------------------------------
#    flat_coords = x_node.reshape(-1, 3)
#    flat_disp = y_disp.reshape(-1, 3)
#    #flat_stress = y_stress.reshape(-1, 6)  # ← 6 components: SX, SY, SZ, SXY, SYZ, SXZ
#
#    coord_scaler = StandardScaler().fit(flat_coords)
#    disp_scaler = StandardScaler().fit(flat_disp)
#    #stress_scaler = StandardScaler().fit(flat_stress)
#
#    # Transform
#    x_scaled = coord_scaler.transform(flat_coords).reshape(N_cases, N_nodes, 3)
#    disp_scaled = disp_scaler.transform(flat_disp).reshape(N_cases, N_nodes, 3)
#    #stress_scaled = stress_scaler.transform(flat_stress).reshape(N_cases, N_nodes, 6)
#
#    # ✅ Save scalers with stress
#    np.savez(
#        "scalers.npz",
#        coord_mean=coord_scaler.mean_,
#        coord_std=coord_scaler.scale_,
#        disp_mean=disp_scaler.mean_,
#        disp_std=disp_scaler.scale_
#        #stress_mean=stress_scaler.mean_,   # ← Now included
#        #stress_std=stress_scaler.scale_    # ← Now included
#    )
#    print("✅ Saved scalers.npz with stress_mean/std")
#
#    # Convert to torch
#    x_torch = torch.tensor(x_scaled, dtype=torch.float32)
#    disp_torch = torch.tensor(disp_scaled, dtype=torch.float32)
#    #stress_torch = torch.tensor(stress_scaled, dtype=torch.float32)
#    angles_torch = torch.tensor(layup_angles, dtype=torch.float32)
#    thick_torch = torch.tensor(layup_thicknesses, dtype=torch.float32)
#    edge_index_torch = torch.tensor(edge_index_np, dtype=torch.long)
#
#    # Create dataset (your existing logic)
#    dataset = []
#    for i in range(N_cases):
#        pos = x_torch[i]
#        u = torch.cat([angles_torch[i], thick_torch[i]], dim=0).unsqueeze(0).expand(N_nodes, -1)
#        x = torch.cat([pos, u], dim=1)
#        y = torch.cat([disp_torch[i]], dim=1)
#
#        graph = Data(
#            x=x,
#            y=y,
#            pos=pos,
#            edge_index=edge_index_torch,
#            plies=plies_list[i],
#            materials=materials_dict,
#            layup_angles=angles_torch[i],
#            layup_thicknesses=thick_torch[i]
#        )
#        dataset.append(graph)
#
#    scalers = {
#        'coord': coord_scaler,
#        'disp': disp_scaler
#    }
#    return dataset, scalers

def load_composite_dataset(dataset_file='composite_dataset.npz'):
    data = np.load(dataset_file, allow_pickle=True)
    
    # Extract arrays
    x_node = data['x']                     # [N_cases, N_nodes, 3]
    y_disp = data['y_disp']                # [N_cases, N_nodes, 3]
    layup_angles = data['layup_angles']
    layup_thicknesses = data['layup_thicknesses']
    edge_index_np = data['edge_index']
    
    # New padded plies (modern format)
    plies_padded = data['plies']           # [N_cases, max_plies, 3]
    plies_mask   = data['plies_mask']      # [N_cases, max_plies] bool

    # --- SAFELY extract materials ---
    mat = data['materials']
    if isinstance(mat, np.ndarray):
        if mat.ndim == 0:
            materials_dict = mat[()]
        elif mat.size == 1:
            materials_dict = mat.item()
        else:
            materials_dict = mat
    else:
        materials_dict = mat

    N_cases, N_nodes, _ = x_node.shape

    # -------------------------------
    # Normalize coordinates & displacements
    # -------------------------------
    flat_coords = x_node.reshape(-1, 3)
    flat_disp = y_disp.reshape(-1, 3)

    coord_scaler = StandardScaler().fit(flat_coords)
    disp_scaler = StandardScaler().fit(flat_disp)

    x_scaled = coord_scaler.transform(flat_coords).reshape(N_cases, N_nodes, 3)
    disp_scaled = disp_scaler.transform(flat_disp).reshape(N_cases, N_nodes, 3)

    np.savez(
        "scalers.npz",
        coord_mean=coord_scaler.mean_,
        coord_std=coord_scaler.scale_,
        disp_mean=disp_scaler.mean_,
        disp_std=disp_scaler.scale_
    )
    print("Saved scalers.npz")

    # Convert to torch
    x_torch = torch.tensor(x_scaled, dtype=torch.float32)
    disp_torch = torch.tensor(disp_scaled, dtype=torch.float32)
    edge_index_torch = torch.tensor(edge_index_np, dtype=torch.long)

    # -------------------------------
    # Build dataset
    # -------------------------------
    dataset = []
    for i in range(N_cases):
        pos = x_torch[i]
        
        # Build per-node layup features (repeat the same for all nodes)
        angles = torch.tensor(layup_angles[i], dtype=torch.float32)
        thicks = torch.tensor(layup_thicknesses[i], dtype=torch.float32)
        layup_feat = torch.cat([angles, thicks], dim=0)  # [2*max_plies]
        layup_feat = layup_feat.unsqueeze(0).expand(N_nodes, -1)  # [N_nodes, 2*max_plies]

        x = torch.cat([pos, layup_feat], dim=1)  # node features
        y = disp_torch[i]                        # target: displacement

        # Extract plies for this graph (with mask)
        plies_this = torch.tensor(plies_padded[i], dtype=torch.float32)
        mask_this = torch.tensor(plies_mask[i], dtype=torch.bool)

        graph = Data(
            x=x,
            y=y,
            pos=pos,
            edge_index=edge_index_torch,
            plies=plies_this,      # [max_plies, 3]
            plies_mask=mask_this,  # [max_plies]
            materials=materials_dict
        )
        dataset.append(graph)

    scalers = {
        'coord': coord_scaler,
        'disp': disp_scaler
    }
    return dataset, scalers
