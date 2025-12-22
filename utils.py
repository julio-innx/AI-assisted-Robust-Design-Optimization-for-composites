from pathlib import Path
import streamlit as st
from conv import generate_apdl_from_json
import numpy as np
import torch
import pandas as pd
import json
from gnn.load_data import load_composite_dataset
from gnn.train import train_gnn
from gnn.model import CompositeGNN
from gnn.predict import load_model_and_scalers
import os
import json
import random
import plotly.graph_objects as go

# utils.py or load_data.py
def parse_materials_from_configs(configs_dir="configs"):
    """
    Parse all config_XXXX.json files to extract materials and plies.

    Returns:
        List of dicts: [
            {'materials': [...], 'plies': [{'angle': ..., 'thickness': ..., 'material_id': ...}, ...]},
            ...
        ]
    """
    configs_dir = Path(configs_dir)
    config_files = sorted(configs_dir.glob("config_*.json"))
    materials_list = []

    for cfg_file in config_files:
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)

        sections = cfg.get('sections', [])
        plies = []  # ← This will store all plies for this case

        # Loop over each section
        for sec in sections:
            # Loop over each ply in the section
            for p in sec['plies']:
                plies.append({
                    'angle': float(p['angle']),
                    'thickness': float(p['thickness']),
                    'material_id': int(p['material_id']),
                    'integration_points': int(p.get('integration_points', 3))
                })

        # Append both materials and processed plies
        materials_list.append({
            'materials': cfg['materials'],  # Full list of material definitions
            'plies': plies                  # List of ply dictionaries
        })

    return materials_list

def read_ansys_table(file_path, col_names, data_start_keyword=None, skip_rows=0):
    """
    Robustly read ANSYS .txt output tables.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Start parsing after keyword or skip rows
    start_line = 0
    if data_start_keyword:
        for i, line in enumerate(lines):
            if data_start_keyword in line:
                start_line = i + 1
                break
    else:
        start_line = skip_rows

    for line in lines[start_line:]:
        stripped = line.strip()
        if not stripped or "NODE" in stripped or "ELEM" in stripped or "LOAD STEP" in stripped:
            continue  # Skip headers
        parts = stripped.split()
        if not parts:
            continue
        try:
            # Try to convert first column to int (node/element ID)
            int(parts[0])
            if len(parts) >= len(col_names):
                data.append([float(x) if '.' in x or 'E' in x.upper() else int(x) for x in parts[:len(col_names)]])
        except ValueError:
            continue  # Not a data row

    return pd.DataFrame(data, columns=col_names)

def parse_all_results(cases_dir="cases", configs_dir="configs", output_file="composite_dataset.npz"):
    cases_dir = Path(cases_dir)
    configs_dir = Path(configs_dir)
    case_dirs = sorted([d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])

    # Lists to store data
    node_coords_list = []
    disp_list = []
    layup_angles_list = []
    layup_thicknesses_list = []

    N_nodes = None
    edge_index = None

    print(f"Parsing {len(case_dirs)} cases...")

    for i, case_dir in enumerate(case_dirs):
        result_dir = case_dir / "results"
        config_path = configs_dir / f"config_{i:04d}.json"

        print(f"Parsing {case_dir.name}...")

        # -------------------------------
        # 1. Load config → get layup
        # -------------------------------
#        with open(config_path, 'r') as f:
#            cfg = json.load(f)
#        sec = cfg['sections'][0]
#        angles = [p['angle'] for p in sec['plies']]
#        thicknesses = [p['thickness'] for p in sec['plies']]
#
#        layup_angles_list.append(angles)
#        layup_thicknesses_list.append(thicknesses)
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        # Extract materials once (same for all cases) – save only on first case
        if i == 0:
            global_materials_dict = {mat['id']: mat for mat in cfg['materials']}

        # Build plies tensor: [num_plies, 3] = [angle, thickness, material_id]
        plies_this_case = []
        for sec in cfg['sections']:                    # now supports multiple sections!
            for ply in sec['plies']:
                plies_this_case.append([
                    float(ply['angle']),
                    float(ply['thickness']),
                    float(ply['material_id'])
                ])

        # Convert to tensor and store per case
        plies_tensor = torch.tensor(plies_this_case, dtype=torch.float32)

        # Also keep old simple lists for backward compatibility (if needed)
        angles = [p['angle'] for p in cfg['sections'][0]['plies']]
        thicknesses = [p['thickness'] for p in cfg['sections'][0]['plies']]
        layup_angles_list.append(angles)
        layup_thicknesses_list.append(thicknesses)
        if 'plies_list' not in locals():
            plies_list = []
        plies_list.append(plies_tensor)

        # -------------------------------
        # 2. Parse node_list.txt → coordinates
        # -------------------------------
        df_nodes = read_ansys_table(
            result_dir / "node_list.txt",
            col_names=['NODE', 'X', 'Y', 'Z', 'THXY', 'THYZ', 'THZX'],
            skip_rows=8
        )

        if df_nodes is None or df_nodes.empty:
            st.error(f"❌ Failed to read node_list for {case_dir}")
            continue

        # ✅ Convert to NumPy array and sort by NODE column (index 0)
        data_array = df_nodes.values           # Now it's a NumPy array: [N, 7]
        sorted_indices = data_array[:, 0].argsort()  # Sort by NODE (first column)
        sorted_data = data_array[sorted_indices]

        coords = sorted_data[:, 1:4]  # X, Y, Z
        node_ids = sorted_data[:, 0].astype(int)

        if N_nodes is None:
            N_nodes = len(coords)
        else:
            assert len(coords) == N_nodes, f"Node count mismatch in {case_dir}"

        node_coords_list.append(coords)

        # -------------------------------
        # 3. Parse disp_list.txt
        # -------------------------------
        df_disp = read_ansys_table(
            result_dir / "disp_list.txt",
            col_names=['NODE', 'UX', 'UY', 'UZ', 'USUM'],
            skip_rows=7
        )
        if df_disp is None or df_disp.empty:
            print(f"❌ Missing/invalid disp in {case_dir}")
            continue

        # Align displacement by node ID
        disp_sorted = np.zeros((N_nodes, 3))
        for _, row in df_disp.iterrows():
            nid = int(row['NODE'])
            idx = np.where(node_ids == nid)[0]
            if len(idx) > 0:
                disp_sorted[idx[0]] = [row['UX'], row['UY'], row['UZ']]
        disp_list.append(disp_sorted)

        # -------------------------------
        # 4. Parse stress_node_comp_list.txt
        # -------------------------------
        df_stress = read_ansys_table(
            result_dir / "stress_node_comp_list.txt",
            col_names=['NODE', 'SX', 'SY', 'SZ', 'SXY', 'SYZ', 'SXZ'],
            skip_rows=7
        )
        if df_stress is None or df_stress.empty:
            print(f"❌ Missing/invalid stress in {case_dir}")
            continue

        stress_sorted = np.zeros((N_nodes, 6))
        for _, row in df_stress.iterrows():
            nid = int(row['NODE'])
            idx = np.where(node_ids == nid)[0]
            if len(idx) > 0:
                stress_sorted[idx[0]] = [
                    row['SX'], row['SY'], row['SZ'],
                    row['SXY'], row['SYZ'], row['SXZ']
                ]
        #stress_comp_list.append(stress_sorted)

        # -------------------------------
        # 5. Build edge_index ONCE from first case
        # -------------------------------
        if edge_index is None:
            elem_file = result_dir / "element_list.txt"
            edges = set()
            with open(elem_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts or not parts[0].isdigit():
                    continue
                try:
                    raw_nodes = list(map(int, parts[-4:]))  # last 4 are nodes
                    nodes = [n - 1 for n in raw_nodes]     # 1-based → 0-based
                    for i in range(len(nodes)):
                        for j in range(i+1, len(nodes)):
                            ni, nj = min(nodes[i], nodes[j]), max(nodes[i], nodes[j])
                            edges.add((ni, nj))
                except Exception as e:
                    continue

            src, dst = zip(*edges) if edges else ([], [])
            edge_index = np.array([src, dst])
            print(f"✅ Built shared edge_index: {edge_index.shape[1]} edges")

    # -------------------------------
    # 6. Pad variable-length layups
    # -------------------------------
#    max_plies = max(len(a) for a in layup_angles_list)
    def pad(seq, length, value=0.0):
        return list(seq) + [value] * (length - len(seq))
#
#    angles_padded = np.array([pad(a, max_plies) for a in layup_angles_list])
#    thick_padded = np.array([pad(t, max_plies) for t in layup_thicknesses_list])
    max_plies = max(len(a) for a in layup_angles_list)
    angles_padded    = np.array([a + [0] * (max_plies - len(a)) for a in layup_angles_list])
    thick_padded     = np.array([t + [0] * (max_plies - len(t)) for t in layup_thicknesses_list])

    max_plies_full = max(len(p) for p in plies_list)
    plies_padded = np.zeros((len(plies_list), max_plies_full, 3))   # [N_cases, max_plies, 3]
    plies_mask   = np.zeros((len(plies_list), max_plies_full), dtype=bool)

    for i, plies_tensor in enumerate(plies_list):
        n = len(plies_tensor)
        plies_padded[i, :n] = plies_tensor.numpy()   # or .cpu().numpy() if on GPU
        plies_mask[i, :n]   = True

    # -------------------------------
    # 7. Save .npz
    # -------------------------------
    X = np.stack(node_coords_list)
    Y_disp = np.stack(disp_list)
    #Y_stress = np.stack(stress_comp_list)
    angles_padded = np.array([pad(a, max_plies) for a in layup_angles_list])
    thick_padded = np.array([pad(t, max_plies) for t in layup_thicknesses_list])

#    np.savez(
#        output_file,
#        x=X,
#        y_disp=Y_disp,
#        layup_angles=angles_padded,
#        layup_thicknesses=thick_padded,
#        edge_index=edge_index
#    )
    np.savez(
        output_file,
        x=X,
        y_disp=Y_disp,
        layup_angles=angles_padded,        # kept for old code
        layup_thicknesses=thick_padded,   # kept for old code
        edge_index=edge_index,
        
        plies=plies_padded,                # [N_cases, max_plies, 3]
        plies_mask=plies_mask,             # True where ply exists
        materials=global_materials_dict,
    )
#     print(f"✅ Dataset saved to {output_file}")
#     print(f"   Shapes:")
#     print(f"     x: {X.shape}")
#     print(f"     y_disp: {y_disp.shape}")
#     print(f"     y_stress_comp: {y_stress_comp.shape}")
#     print(f"     layup_angles: {layup_angles.shape}")
#     print(f"     edge_index: {edge_index.shape}")

    return output_file


# parse_dlist.py
def parse_dlist(input_file, output_file):
    """
    Parse ANSYS DLIST output and create a clean table:
        NODE  UX  UY  UZ  ROTX  ROTY  ROTZ
    """
    # Initialize storage
    bc_data = {}

    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip headers
        if "LIST CONSTRAINTS" in line or "CURRENTLY SELECTED DOF" in line:
            continue
        if "NODE" in line and "LABEL" in line and "REAL" in line:
            continue
        if "IMAG" in line:  # Another header line
            continue

        # Split line (multiple spaces)
        parts = [p for p in line.split() if p]

        # Must have at least 4 columns: NODE, LABEL, REAL, IMAG
        if len(parts) >= 4:
            try:
                node = int(parts[0])
                label = parts[1]
                value = float(parts[2])  # Real part

                # Initialize node if not seen
                if node not in bc_data:
                    bc_data[node] = {'UX': 0.0, 'UY': 0.0, 'UZ': 0.0,
                                     'ROTX': 0.0, 'ROTY': 0.0, 'ROTZ': 0.0}

                # Store value
                if label in bc_data[node]:
                    bc_data[node][label] = value
            except ValueError:
                # Skip malformed lines
                continue

    # Write to output file
    with open(output_file, 'w') as f:
        f.write("NODE UX UY UZ ROTX ROTY ROTZ\n")
        for node in sorted(bc_data.keys()):
            d = bc_data[node]
            f.write(f"{node:4d} {d['UX']:1.0f} {d['UY']:1.0f} {d['UZ']:1.0f} "
                    f"{d['ROTX']:1.0f} {d['ROTY']:1.0f} {d['ROTZ']:1.0f}\n")

    print(f"✅ Parsed DLIST saved to {output_file}")
    print(f"   {len(bc_data)} nodes with boundary conditions")


def parse_element_table(input_file, output_file):
    """
    Parse ANSYS ELIST output and extract:
        ELEM, MAT, TYP, REL, ESY, SEC, N1, N2, N3, N4
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    data = []
    header_written = False

    for line in lines:
        line = line.strip()
        # Skip empty or irrelevant lines
        if not line or "SELECT" in line or "LIST" in line or "NODES" in line:
            continue

        # Split line into parts (handles multiple spaces)
        parts = [p for p in line.split() if p]

        # Must have at least 10 columns: 6 element attrs + 4 nodes
        if len(parts) >= 10:
            try:
                # Parse element and attributes
                elem = int(parts[0])
                mat = int(parts[1])
                typ = int(parts[2])
                rel = int(parts[3])
                esy = int(parts[4])
                sec = int(parts[5])
                n1 = int(parts[6])
                n2 = int(parts[7])
                n3 = int(parts[8])
                n4 = int(parts[9])

                # Write header only once
                if not header_written:
                    data.append("ELEM MAT TYP REL ESY SEC N1 N2 N3 N4")
                    header_written = True

                # Append data row
                data.append(f"{elem:4d} {mat:3d} {typ:3d} {rel:3d} {esy:3d} {sec:3d} {n1:4d} {n2:4d} {n3:4d} {n4:4d}")
            except ValueError:
                # Skip lines that can't be parsed
                continue

    # Write to output file
    with open(output_file, 'w') as f:
        for line in data:
            f.write(line + '\n')

    print(f"✅ Parsed data saved to {output_file}")


# parse_plist.py
def parse_plist(input_file, output_file):
    """
    Parse ANSYS PLIST output (nodal surface loads) and create:
        ELEM  FACE  N1  N2  N3  N4  PRESSURE
    """
    data = []
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip headers
        if "LIST NODAL SURFACE LOAD" in line or "STEP" in line:
            continue
        if "ELEMENT" in line and "FACE" in line and "NODES" in line:
            continue
        if "IMAGINARY" in line:
            continue

        parts = [p for p in line.split() if p]

        # Case 1: Full line with ELEM, FACE, N1, N2, N3, N4, PRES
        if len(parts) >= 8 and parts[0].isdigit():
            try:
                elem = int(parts[0])
                face = int(parts[1])
                n1 = int(parts[2])
                n2 = int(parts[3])
                n3 = int(parts[4]) if parts[4] != '0' else 0
                n4 = int(parts[5]) if parts[5] != '0' else 0
                pressure = float(parts[6])
                data.append((elem, face, n1, n2, n3, n4, pressure))
            except ValueError:
                continue

        # Case 2: Continuation line (only pressure, same as previous)
        elif len(parts) >= 2 and '.' in parts[0]:
            try:
                # Reuse last element, just update pressure (same face)
                if data:
                    prev = data[-1]
                    data.append((prev[0], prev[1], prev[2], prev[3], prev[4], prev[5], float(parts[0])))
            except ValueError:
                continue

    # Write to output
    with open(output_file, 'w') as f:
        f.write("ELEM FACE N1 N2 N3 N4 PRESSURE\n")
        for row in data:
            f.write(f"{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d} {row[5]:4d} {row[6]:8.3f}\n")

    print(f"✅ Parsed PLIST saved to {output_file}")
    print(f"   {len(data)} pressure-loaded faces")

def parse_ansys_table(filename):
    """Robust parser for ANSYS whitespace-separated tables"""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or any(kw in line.upper() for kw in ['SELECT', 'LIST', 'NODES', 'ELEM']):
                continue
            parts = [p for p in line.split() if p]
            if len(parts) >= 10:
                try:
                    row = [int(x) if i < 10 else float(x) for i, x in enumerate(parts)]
                    data.append(row)
                except ValueError:
                    continue
    return data

# ===============================
# DATABASE GENERATION FUNCTION
# ===============================
def generate_laminate_database(n_configs, config_dir="configs"):
    """
    Generate N random laminate configurations using current GUI settings.
    Preserves geometry, materials, mesh, BCs, loads, attributes.
    """
    os.makedirs(config_dir, exist_ok=True)

    # Build base config from current Streamlit state
    base_config = {
        "geometry": {
            "source": "iges",
            "filename": st.session_state.get('file_name', 'plate'),
            "path": st.session_state.get('path', 'Downloads\\'),
            "scale_factor": float(st.session_state.get('scale_factor', 100.0))
        },
        "materials": st.session_state['materials'],
        "mesh": {
            "element_size": float(st.session_state['element_size']),
            "shape": st.session_state['shape'],
            "mesh_type": st.session_state['mesh_type']
        },
        "attributes": st.session_state['attributes'],
        "boundary_conditions": st.session_state['boundary_conditions'],
        "loads": st.session_state['loads'],
        "output": {
            "export_displacement": st.session_state.get('export_disp', True),
            "export_stress_components": st.session_state.get('export_stress_comp', True),
            "export_principal_stress": st.session_state.get('export_principal', True)
        }
    }

    db_params = st.session_state['db_params']
    angle_options = st.session_state['db_angle_options']

    for i in range(n_configs):
        secs = []
        attrs = []

        for sid in range(1, db_params['n_sections'] + 1):
            n_ply = random.randint(db_params['min_plies'], db_params['max_plies'])
            plies = []
            for _ in range(n_ply):
                plies.append({
                    # "thickness": round(random.uniform(db_params['thickness_min'], db_params['thickness_max']), 3),
                    "thickness": max(0.05, round(random.uniform(db_params['thickness_min'], db_params['thickness_max']), 3)),
                    "material_id": random.randint(1, len(base_config["materials"])),
                    "angle": random.choice(angle_options),
                    "integration_points": 3
                })
            secs.append({
                "id": sid,
                "type": "SHELL",
                "plies": plies,
                "offset": db_params['common_offset']
            })
            # Example area mapping: Area 3 → Sec 1, Area 4 → Sec 2
            attrs.append({"area_id": 3 if sid == 1 else 4, "section_id": sid})

        config = base_config.copy()
        config["sections"] = secs
        config["attributes"] = attrs

        with open(f"{config_dir}/config_{i:04d}.json", 'w') as f:
            json.dump(config, f, indent=2)

    return n_configs

def generate_all_cases(config_dir="configs", cases_dir="cases"):
    """
    Loop over all config_xxx.json and create case folders.
    """
    config_dir = Path(config_dir)
    cases_dir = Path(cases_dir)
    progress_bar = st.progress(0)
    status_text = st.empty()

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    cases_dir.mkdir(exist_ok=True)
    config_files = sorted(config_dir.glob("config_*.json"))

    if not config_files:
        raise FileNotFoundError(f"No config_*.json files in {config_dir}")

    print(f"Found {len(config_files)} configs")

    for i, config_path in enumerate(config_files):
        case_name = f"case_{i+1:03d}"
        case_folder = cases_dir / case_name

        print(f"[{i+1}/{len(config_files)}] Generating {case_name}")
        if case_folder.exists():
            status_text.text(f"Skipped (exists): {case_name}")
            progress_bar.progress((i + 1) / len(config_files))
            continue
        try:
            generate_apdl_from_json(
                config_path=config_path,
                case_dir=case_folder,
                config_file_name="config.json"
            )
        except Exception as e:
            print(f"❌ Failed on {config_path.name}: {e}")
            continue
        progress_bar.progress((i + 1) / len(config_files))

    print(f"🎉 Generated {len(config_files)} cases in {cases_dir}/")

    st.success(f"🎉 Generated {len(config_files)} cases in `{cases_dir}/`")
    st.balloons()

# ===============================
# TRAIN GNN TAB
# ===============================

def train_button():
    st.header("🧠 Train GNN on Simulation Data")

    # Initialize session state
    if 'training_active' not in st.session_state:
        st.session_state.training_active = False

    # Auto-detect resume point
    start_epoch = 0
    loss_history = []

    # Look for saved models
    if os.path.exists("checkpoints/composite_gnn_interrupt.pth"):
        ckpt = torch.load("checkpoints/composite_gnn_interrupt.pth", map_location='cpu')
        start_epoch = ckpt['epoch']
        loss_history = ckpt['loss_history']
        st.info(f"🔁 Found checkpoint at epoch {start_epoch} — click 'Start' to resume")
    elif os.path.exists("composite_gnn.pth") and not st.session_state.training_active:
        try:
            # Try to load metadata
            from copy import deepcopy
            ckpt = torch.load("composite_gnn.pth", map_location='cpu')
            if isinstance(ckpt, dict) and 'epoch' in ckpt:
                start_epoch = ckpt['epoch']
                loss_history = ckpt.get('loss_history', [])
                st.info(f"🔁 Last saved model trained up to epoch {start_epoch}")
        except:
            pass

    st.subheader("Hyperparameters")
    tolerance = st.number_input("Convergence Tolerance", min_value=1e-7, value=1e-5, format="%.2e")
    max_epochs = st.number_input("Max Epochs", min_value=50, value=1000)
    lr = st.number_input("Learning Rate", min_value=1e-7, value=3e-5, format="%.6f")
    hidden = st.number_input("Hidden Units", min_value=16, value=64, step=16)

    col1, col2 = st.columns(2)

    # Start Button
    if col1.button("🚀 Start / Resume Training", disabled=st.session_state.training_active):
        st.session_state.training_active = True
        try:
            from gnn.load_data import load_composite_dataset
            from gnn.train import train_gnn
            from gnn.model import CompositeGNN

            st.info("Loading dataset...")

            dataset, scalers = load_composite_dataset("composite_dataset.npz")
            valid_dataset = [g for g in dataset if not torch.isnan(g.y).any()]
            if not valid_dataset:
                st.error("❌ No valid graphs found!")
                st.session_state.training_active = False
                return

            st.success(f"✅ Loaded {len(valid_dataset)} cases")

            # Progress widgets
            progress_bar = st.progress(start_epoch / max_epochs)
            loss_chart = st.empty()
            if loss_history:
                loss_chart.line_chart(loss_history)

            materials_dict = valid_dataset[0].materials

            model, hist = train_gnn(
                dataset=valid_dataset,
                tolerance=tolerance,
                max_epochs=max_epochs,
                lr=lr,
                hidden=hidden,
                progress_bar=progress_bar,
                loss_chart=loss_chart,
                start_epoch=start_epoch,
                loss_history=loss_history.copy(),
                materials_dict=materials_dict  
            )

            # After training completes
            st.session_state.training_active = False

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            st.exception(e)
            st.session_state.training_active = False

    # Stop Button
    if col2.button("🛑 Stop Training", disabled=st.session_state.training_active):
        from gnn.train import set_stop_training
        set_stop_training()
        st.warning("🛑 Stopping requested... will stop after current batch.")

    # Show status
    if st.session_state.training_active:
        st.info("⚡ Training in progress...")
    else:
        if start_epoch > 0:
            st.success(f"✅ Ready to resume from epoch {start_epoch}")
        else:
            st.info("🟢 Click 'Start' to begin training")

def prediction_button():
    st.header("🔮 Predict Stress & Displacement")

    # Always try to load model when predicting
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    st.write("Define laminate layup:")

    n_plies = st.number_input("Number of plies", min_value=1, max_value=10, value=4, step=1)
    angles = []
    thicknesses = []

    col1, col2 = st.columns(2)
    for i in range(n_plies):
        with col1:
            angle = st.number_input(f"Ply {i+1} Angle (°)", value=float(45 if i % 2 == 0 else -45), key=f"ang_{i}")
            angles.append(angle)
        with col2:
            thick = st.number_input(f"Ply {i+1} Thickness (mm)", value=0.15, step=0.01, key=f"thk_{i}")
            thicknesses.append(thick)

    if st.button("🚀 Predict Field"):
        try:
            from gnn.predict import predict_laminate, load_model_and_scalers

            # ✅ Always force reload before prediction
            success = load_model_and_scalers()

            if not success:
                st.error("❌ Failed to load model/scalers. Did you train?")
                st.warning("💡 Required files:")
                st.code("composite_gnn.pth\nscalers.npz\ncomposite_dataset.npz")
                return

            st.session_state.model_loaded = True
            st.success("✅ Model loaded!")

            with st.spinner("Running GNN inference..."):
                disp, coords = predict_laminate(angles, thicknesses)
            st.success("🎉 Prediction complete!")

            # Compute derived fields
            U_mag = np.linalg.norm(disp, axis=1)

            # ===============================
            # PLOT DISPLACEMENT
            # ===============================
            st.subheader("Displacement Field")
            fig_disp = go.Figure()
            fig_disp.add_trace(go.Scatter3d(
                x=coords[:,0], y=coords[:,1], z=coords[:,2],
                mode='markers',
                marker=dict(size=4, color=U_mag, colorscale='Viridis', showscale=True),
                text=[f"UX={u[0]:.4f}<br>UY={u[1]:.4f}<br>UZ={u[2]:.4f}" for u in disp],
                hoverinfo="text"
            ))
            fig_disp.update_layout(
                title="Displacement Magnitude",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                height=600
            )
            st.plotly_chart(fig_disp, use_container_width=True)

            # ===============================
            # DOWNLOAD RESULTS
            # ===============================
            df_results = pd.DataFrame({
                'X': coords[:,0], 'Y': coords[:,1], 'Z': coords[:,2],
                'UX': disp[:,0], 'UY': disp[:,1], 'UZ': disp[:,2]
            })

            st.download_button(
                "💾 Download Results (CSV)",
                df_results.to_csv(index=False),
                "predicted_results.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)
