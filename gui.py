# gui_composite.py
import streamlit as st
import json
import os
from pathlib import Path
from control import setup
from utils import *
import numpy as np
from master import generate_master_batch
import subprocess
from utils import parse_all_results
import subprocess
import plotly.graph_objects as go
import pandas as pd
            
            

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(layout="wide", page_title="FiberNet Designer")
# Custom CSS for sticky header
# st.markdown("""
#     <style>
#         /* Fix top header or first container */
#         .stContainer {
#             position: fixed;
#             top: 0;
#         }
#     </style>
# """, unsafe_allow_html=True)
# with st.container():
st.title("🎯 FiberNet Designer: Composite GUI Studio")
st.write("Define composites interactively. Generate `config.json` or batch datasets.")
setup_tab, simulation_tab, train_tab, test_tab = st.tabs(
    ["Setup", "Simulation", "Training", "Model"]
)

with setup_tab:
    setup()

with simulation_tab:
    st.header("📁 Generate All Simulation Cases")

    if st.button("📦 Build Case Folders for All Configs"):
        if not Path("configs").exists():
            st.error("❌ Run 'Generate Database' first.")
        else:
            generate_all_cases(config_dir="configs", cases_dir="cases")

    st.header("🏁 Run Full Simulation Batch")

    if st.button("🚀 Generate Master Script & Run ANSYS"):
        try:
            with st.spinner("Generating master_batch.mac..."):
                generate_master_batch(config_dir="configs", output_file="master_batch.mac")
            st.success("✅ master_batch.mac generated!")

            # Now run ANSYS
            ansys_exe = r"/opt/ansys_inc/v242/ansys/bin/mapdl"
            mac_file = "master_batch.mac"
            log_file = "master_output.out"

            if not Path(ansys_exe).exists():
                st.warning("⚠️ ANSYS executable not found. Using PATH fallback.")
                ansys_exe = r"/opt/ansys_inc/v242/ansys/bin/mapdl"

            cmd = [str(ansys_exe), "-b", "-i", mac_file, "-o", log_file]

            with st.spinner("Running ANSYS... This may take a while."):
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    st.success("🎉 ANSYS simulation completed!")
                    st.info("Check `master_output.out` and `/cases/*/results/` for data.")
                else:
                    st.error("❌ ANSYS failed. Check error below.")
                    st.code(result.stderr)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

    # ===============================
    # FINAL: PARSE RESULTS
    # ===============================
    st.header("📊 Parse Simulation Results")

    if st.button("🧠 Build Dataset for GNN"):
        try:
            with st.spinner("Parsing all results into dataset..."):
                dataset_file = parse_all_results(
                    cases_dir="cases",
                    configs_dir="configs",
                    output_file="composite_dataset.npz"
                )
            
            st.success("✅ Dataset built!")
            st.balloons()

            # Show preview
            data = np.load(dataset_file, allow_pickle=True)
            st.write("### Dataset Preview")
            data = np.load(dataset_file, allow_pickle=True)
            
            for k, v in data.items():
                if isinstance(v, np.ndarray):
                    if v.dtype == object:
                        # Handle both scalar object arrays and [dict] arrays
                        if v.ndim == 0:
                            # 0-dim: v[()] gives the actual dict
                            st.json(v[()])
                        elif v.size == 1:
                            st.json(v[0])
                        else:
                            st.text(f"{k}: object array with {v.size} elements")
                    else:
                        st.text(f"{k}: {v.shape}  (dtype={v.dtype})")
                else:
                    st.text(f"{k}: {type(v)} = {v}")
#            st.write("### Dataset Preview")
#            for k, v in data.items():
#                if isinstance(v, np.ndarray) and v.dtype == object:
#                    st.json(v[0])  # show dict
#                elif isinstance(v, np.ndarray):
#                    st.text(f"{k}: {v.shape}")
#                else:
#                    st.text(f"{k}: {v}")

            # Download
            with open(dataset_file, 'rb') as f:
                st.download_button("⬇️ Download Dataset", f, "composite_dataset.npz", "application/octet-stream")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

    # ===============================
    # 🔍 SEARCH & INSPECT CASES
    # ===============================
    st.header("🔍 Search Simulation Cases")

# Ensure cases exist
    cases_dir = Path("cases")
    if not cases_dir.exists():
        st.warning("No cases found. Generate cases first.")
    else:
        case_folders = sorted([d for d in cases_dir.iterdir() if d.is_dir() and d.name.startswith("case_")])
        if not case_folders:
            st.warning("No case folders found in `cases/`")
        else:
            st.write(f"Found {len(case_folders)} cases.")

            # --- Search Options ---
            search_type = st.radio(
                "Search by:",
                ["Case ID", "Ply Angles", "Load Value", "Section ID"],
                horizontal=True
            )

            matching_cases = []

            if search_type == "Case ID":
                case_id_input = st.text_input("Enter Case ID (e.g., 1, 001, case_001)", value="1")
                if case_id_input:
                    # Normalize input
                    if case_id_input.isdigit():
                        target_id = int(case_id_input)
                        target_name = f"case_{target_id:03d}"
                    elif case_id_input.startswith("case_"):
                        target_name = case_id_input
                    else:
                        target_name = f"case_{int(case_id_input):03d}"
                    matching_cases = [c for c in case_folders if c.name == target_name]

            elif search_type == "Ply Angles":
                angle_input = st.text_input("Enter angle(s) to match (comma-separated, e.g., 0,45,-45,90)")
                if angle_input:
                    try:
                        target_angles = [int(a.strip()) for a in angle_input.split(",")]
                        for case_dir in case_folders:
                            config_path = case_dir / "config.json"
                            if config_path.exists():
                                with open(config_path) as f:
                                    cfg = json.load(f)
                                # Get angles from first section (extend if needed)
                                angles = [ply["angle"] for ply in cfg["sections"][0]["plies"]]
                                if set(target_angles).issubset(set(angles)):
                                    matching_cases.append(case_dir)
                    except Exception as e:
                        st.error(f"Invalid angle input: {e}")

            elif search_type == "Load Value":
                load_val = st.number_input("Pressure Load (MPa)", value=-500.0, step=10.0)
                for case_dir in case_folders:
                    config_path = case_dir / "config.json"
                    if config_path.exists():
                        with open(config_path) as f:
                            cfg = json.load(f)
                        loads = cfg.get("loads", [])
                        if any(abs(load.get("value", 0) - load_val) < 1e-3 for load in loads):
                            matching_cases.append(case_dir)

            elif search_type == "Section ID":
                sec_id = st.number_input("Section ID", min_value=1, value=1)
                for case_dir in case_folders:
                    config_path = case_dir / "config.json"
                    if config_path.exists():
                        with open(config_path) as f:
                            cfg = json.load(f)
                        if any(sec.get("id") == sec_id for sec in cfg.get("sections", [])):
                            matching_cases.append(case_dir)

            # --- Display Results ---
            if matching_cases:
                st.success(f"✅ Found {len(matching_cases)} matching case(s):")
                for case_dir in matching_cases:
                    with st.expander(f"📁 {case_dir.name}"):
                        config_path = case_dir / "config.json"
                        if config_path.exists():
                            with open(config_path) as f:
                                cfg = json.load(f)
                            st.json(cfg)

                        # --- Action Buttons ---
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("📥 Download Config", key=f"dl_cfg_{case_dir.name}"):
                                with open(config_path, "rb") as f:
                                    st.download_button(
                                        label="Download",
                                        data=f,
                                        file_name=f"{case_dir.name}_config.json",
                                        mime="application/json"
                                    )
                        with col2:
                            if st.button("▶️ Run ANSYS", key=f"run_ansys_{case_dir.name}"):
                                # Run ANSYS on this case only
                                try:
                                    ansys_exe = r"/opt/ansys_inc/v242/ansys/bin/mapdl"
                                    main_script = case_dir / "main_script.mac"
                                    log_file = case_dir / "ansys_output.log"
                                    with st.spinner(f"Running ANSYS on {case_dir.name}..."):
                                        result = subprocess.run(
                                            [ansys_exe, "-b", "-i", str(main_script), "-o", str(log_file)],
                                            cwd=case_dir,
                                            capture_output=True,
                                            text=True,
                                            timeout=300
                                        )
                                    if result.returncode == 0:
                                        st.success("✅ ANSYS finished!")
                                    else:
                                        st.error("❌ ANSYS failed.")
                                        st.text_area("Log", result.stderr[-1000:], height=150)
                                except Exception as e:
                                    st.exception(e)
                        with col3:
                            results_dir = case_dir / "results"
                            if (results_dir / "node_disp.txt").exists():
                                if st.button("⬇️ Download Results", key=f"dl_res_{case_dir.name}"):
                                    # Reuse your download logic here
                                    df_node = pd.read_csv(results_dir / "node_disp.txt", delim_whitespace=True, skiprows=1,
                                                          names=["NODE", "X", "Y", "Z", "DX", "DY", "DZ"])
                                    df_stress = pd.read_csv(results_dir / "stress_components.txt", delim_whitespace=True, skiprows=1,
                                                            names=["NODE", "SX", "SY", "SZ", "SXY", "SYZ", "SXZ"])
                                    df = pd.merge(df_node, df_stress, on="NODE")
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="Download CSV",
                                        data=csv,
                                        file_name=f"{case_dir.name}_results.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info("No results yet. Run ANSYS first.")
            else:
                st.info("No cases match your search criteria.")

with train_tab:
    train_button()

with test_tab:
    prediction_button()
