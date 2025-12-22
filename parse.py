import numpy as np
from pathlib import Path
import json
import pandas as pd

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
    """
    Parse all ANSYS results into a structured ML-ready dataset.
    """
    cases_dir = Path(cases_dir)
    configs_dir = Path(configs_dir)
    case_dirs = sorted([d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
    
    if not case_dirs:
        raise FileNotFoundError(f"No case folders found in {cases_dir}")

    # Lists to store data
    node_coords_list = []
    disp_list = []
    stress_comp_list = []
    stress_prin_list = []
    layup_angles_list = []
    layup_thicknesses_list = []
    section_offsets = []

    # Assume constant mesh (same node count and order)
    N_nodes = None
    edge_index = None

    print(f"Parsing {len(case_dirs)} cases...")

    for i, case_dir in enumerate(case_dirs):
        result_dir = case_dir / "results"
        config_path = configs_dir / f"config_{i:04d}.json"

        print(f"Parsing {case_dir.name}...")

        # -------------------------------
        # 1. Load config.json → get layup
        # -------------------------------
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        sec = cfg['sections'][0]  # assume primary section
        angles = [ply['angle'] for ply in sec['plies']]
        thicknesses = [ply['thickness'] for ply in sec['plies']]
        offset = sec.get('offset', 'MID')

        max_plies = 10
        angles += [0] * (max_plies - len(angles))
        thicknesses += [0.0] * (max_plies - len(thicknesses))

        layup_angles_list.append(angles)
        layup_thicknesses_list.append(thicknesses)
        section_offsets.append(1 if offset == 'MID' else 0)

        # -------------------------------
        # 2. Parse node_list.txt → coordinates
        # -------------------------------
        node_file = result_dir / "node_list.txt"
        df_nodes = read_ansys_table(
            node_file,
            col_names=['NODE', 'X', 'Y', 'Z', 'THXY', 'THYZ', 'THZX']
        )
        df_nodes.set_index('NODE', inplace=True)
        df_nodes.sort_index(inplace=True)
        coords = df_nodes[['X', 'Y', 'Z']].values
        if N_nodes is None:
            N_nodes = coords.shape[0]
        assert coords.shape[0] == N_nodes, f"Node count mismatch in {case_dir}"
        node_coords_list.append(coords)

        # -------------------------------
        # 3. Parse disp_list.txt → displacement
        # -------------------------------
        disp_file = result_dir / "disp_list.txt"
        df_disp = read_ansys_table(
            disp_file,
            col_names=['NODE', 'UX', 'UY', 'UZ', 'USUM']
        )
        df_disp.set_index('NODE', inplace=True)
        df_disp = df_disp.reindex(df_nodes.index)  # Align
        disp = df_disp[['UX', 'UY', 'UZ']].values
        disp_list.append(disp)
        print(disp_list)

        # -------------------------------
        # 4. Parse stress_node_comp_list.txt
        # -------------------------------
        stress_comp_file = result_dir / "stress_node_comp_list.txt"
        df_stress_comp = read_ansys_table(
            stress_comp_file,
            col_names=['NODE', 'SX', 'SY', 'SZ', 'SXY', 'SYZ', 'SXZ']
        )
        df_stress_comp.set_index('NODE', inplace=True)
        df_stress_comp = df_stress_comp.reindex(df_nodes.index)
        stress_comp = df_stress_comp[['SX', 'SY', 'SZ', 'SXY', 'SYZ', 'SXZ']].values
        stress_comp_list.append(stress_comp)

        # -------------------------------
        # 5. Parse stress_node_prin_list.txt
        # -------------------------------
        stress_prin_file = result_dir / "stress_node_prin_list.txt"
        df_stress_prin = read_ansys_table(
            stress_prin_file,
            col_names=['NODE', 'S1', 'S2', 'S3', 'SINT', 'SEQV']
        )
        df_stress_prin.set_index('NODE', inplace=True)
        df_stress_prin = df_stress_prin.reindex(df_nodes.index)
        stress_prin = df_stress_prin[['S1', 'S2', 'S3', 'SINT', 'SEQV']].values
        stress_prin_list.append(stress_prin)

        # -------------------------------
        # 6. Build edge_index from element_list.txt
        # -------------------------------
        if edge_index is None:
            elem_file = result_dir / "element_list.txt"
            edges = set()
            with open(elem_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0].isdigit() and len(parts) > 4 and 'ELEM' not in line and 'MAT' not in line:
                    try:
                        nodes = list(map(int, parts[-4:]))  # Last 4 are nodes
                        for i in range(len(nodes)):
                            for j in range(i+1, len(nodes)):
                                edges.add((nodes[i], nodes[j]))
                    except:
                        continue

            if edges:
                src, dst = zip(*edges)
                edge_index = np.array([src, dst])

    # Stack into arrays
    X = np.stack(node_coords_list)           # [N_cases, N_nodes, 3]
    y_disp = np.stack(disp_list)             # [N_cases, N_nodes, 3]
    y_stress_comp = np.stack(stress_comp_list)  # [N_cases, N_nodes, 6]
    y_stress_prin = np.stack(stress_prin_list)  # [N_cases, N_nodes, 5]

    layup_angles = np.array(layup_angles_list)        # [N_cases, max_plies]
    layup_thicknesses = np.array(layup_thicknesses_list)  # [N_cases, max_plies]
    section_offset = np.array(section_offsets)        # [N_cases]

    max_plies = max(len(a) for a in layup_angles_list)
    angles_padded = np.array([pad(a, max_plies) for a in layup_angles_list])
    thick_padded = np.array([pad(t, max_plies) for t in layup_thicknesses_list])

    np.savez(
        output_file,
        x=np.stack(node_coords_list),
        y_disp=np.stack(disp_list),
        y_stress_comp=np.stack(stress_comp_list),
        layup_angles=angles_padded,
        layup_thicknesses=thick_padded,
        edge_index=edge_index
    )

    print(f"✅ Dataset saved to {output_file}")
    print(f"   Shapes:")
    print(f"     x: {X.shape}")
    print(f"     y_disp: {y_disp.shape}")
    print(f"     y_stress_comp: {y_stress_comp.shape}")
    print(f"     layup_angles: {layup_angles.shape}")
    print(f"     edge_index: {edge_index.shape}")

    return output_file


#parse_all_results()
