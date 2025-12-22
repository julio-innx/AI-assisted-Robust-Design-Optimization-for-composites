import streamlit as st
import json
import os
from pathlib import Path
import random
from conv import generate_apdl_from_json
from build import build_gnn_dataset
from utils import *

def setup():
    # ===============================
    # 1. GEOMETRY
    # ===============================
    st.header("📐 Geometry")
    col1, col2, col3 = st.columns(3)
    with col1:
        geom_source = st.radio("Source", ["IGES"], help="Geometry type")
    with col2:
        scale_factor = st.number_input("Scale Factor", 1.0, 1000.0, 100.0, step=1.0, format="%.1f")
    with col3:
        file_name = st.text_input("IGES File Name", "plate")

    if geom_source == "IGES":
        path = st.text_input("IGES Path", "")

# ===============================
# 2. MATERIALS
# ===============================
    st.header("🧪 Materials")
    n_materials = st.number_input("Number of Materials", 1, 5, 1, key="n_mats")

    materials = []
    for i in range(n_materials):
        with st.expander(f"Material {i+1}"):
            cols = st.columns([2, 2, 2])
            with cols[0]:
                mat_name = st.text_input("Name", f"CarbonEpoxy_{i+1}", key=f"mat_name_{i}")
            with cols[1]:
                ex = st.number_input("EX (MPa)", 1000.0, 200000.0, 149000.0, step=1000.0, format="%.1f", key=f"ex_{i}")
            with cols[2]:
                ey = st.number_input("EY (MPa)", 1000.0, 100000.0, 8500.0, step=100.0, format="%.1f", key=f"ey_{i}")

            cols = st.columns(4)
            with cols[0]:
                przx = st.number_input("PRXY", 0.0, 0.5, 0.3, step=0.01, key=f"prxy_{i}")
            with cols[1]:
                gxy = st.number_input("GXY (MPa)", 1000.0, 100000.0, 5110.0, step=100.0, format="%.1f", key=f"gxy_{i}")
            with cols[2]:
                gxz = st.number_input("GXZ (MPa)", 1000.0, 100000.0, 5110.0, step=100.0, format="%.1f", key=f"gxz_{i}")
            with cols[3]:
                gyz = st.number_input("GYZ (MPa)", 1000.0, 100000.0, 5110.0, step=100.0, format="%.1f", key=f"gyz_{i}")

            materials.append({
                "id": i+1,
                "name": mat_name,
                "EX": float(ex),
                "EY": float(ey),
                "EZ": float(ey),
                "PRXY": float(przx),
                "PRYZ": 0.35,
                "PRXZ": 0.35,
                "GXY": float(gxy),
                "GYZ": float(gyz),
                "GXZ": float(gxz)
            })

    # ===============================
    # 3. SECTIONS & LAMINATES (TABS)
    # ===============================
    st.header("🧱 Laminate Design")

    tab_manual, tab_db_setup = st.tabs(["🔧 Define Laminate", "⚡ Orientation Setup"])

    sections = []

# -------------------------------
# TAB 1: Manual Laminate Definition
# -------------------------------
    with tab_manual:
        n_sections = st.number_input("Number of Sections", 1, 5, 2, key="n_secs_manual")
        
        for sec_id in range(1, n_sections + 1):
            with st.expander(f"Section {sec_id}"):

                if f'plies_sec{sec_id}' not in st.session_state:
                    st.session_state[f'plies_sec{sec_id}'] = [
                        {"thickness": 0.191, "material_id": 1, "angle": 0}
                    ]

                plies = st.session_state[f'plies_sec{sec_id}']

                st.markdown("**Laminate Plies**")

                for idx, ply in enumerate(plies):
                    c1, c2, c3, c4, c6, c7 = st.columns([1, 1, 1, 1, 1, 1])

                    with c1:
                        st.write(f"Ply {idx+1}")

                    with c2:
                        new_angle = st.number_input(
                            "Angle (deg)",
                            min_value=-180.0,
                            max_value=180.0,
                            value=float(ply["angle"]),
                            step=1.0,
                            format="%.1f",
                            key=f"angle_{sec_id}_{idx}"
                        )
                        st.session_state[f'plies_sec{sec_id}'][idx]["angle"] = new_angle

                    with c3:
                        new_thickness = st.number_input(
                            "Thick (mm)",
                            min_value=0.05,
                            max_value=0.5,
                            value=float(ply["thickness"]),
                            step=0.001,
                            format="%.3f",
                            key=f"t_{sec_id}_{idx}"
                        )
                        st.session_state[f'plies_sec{sec_id}'][idx]["thickness"] = new_thickness

                    with c4:
                        new_mat_id = st.selectbox(
                            "Mat ID",
                            options=list(range(1, n_materials + 1)),
                            index=int(ply["material_id"]) - 1 if 1 <= ply["material_id"] <= n_materials else 0,
                            key=f"matid_{sec_id}_{idx}"
                        )
                        st.session_state[f'plies_sec{sec_id}'][idx]["material_id"] = new_mat_id

                    with c6:
                        st.write("")
                        if len(plies) > 1:
                            if st.button("🗑️", key=f"del_{sec_id}_{idx}"):
                                st.session_state[f'plies_sec{sec_id}'].pop(idx)
                                st.rerun()

                    with c7:
                        st.write("")
                        if idx == len(plies) - 1:
                            if st.button("➕", key=f"add_{sec_id}_{idx}"):
                                st.session_state[f'plies_sec{sec_id}'].append({
                                    "thickness": 0.191,
                                    "material_id": 1,
                                    "angle": 0
                                })
                                st.rerun()

                st.markdown("---")
                section_offset = st.radio(
                    f"Offset - Sec {sec_id}",
                    ["TOP", "MID", "BOT"],
                    index=1,
                    horizontal=True,
                    key=f"sec_offset_{sec_id}"
                )

                sections.append({
                    "id": sec_id,
                    "type": "SHELL",
                    "plies": [
                        {
                            "thickness": float(p['thickness']),
                            "material_id": int(p['material_id']),
                            "angle": int(round(p['angle'])),
                            "integration_points": 3
                        } for p in st.session_state[f'plies_sec{sec_id}']
                    ],
                    "offset": section_offset
                })

        # Save for later use
        st.session_state['manual_sections'] = sections


# -------------------------------
# TAB 2: Database Generation Setup
# -------------------------------
    with tab_db_setup:
        st.markdown("### 🔧 Generate Orientation")

        n_sections_db = st.number_input("Number of Sections", 1, 5, 2, key="n_secs_db")

        angle_input = st.text_input("Ply Angles (comma-separated)", "0, 45, -45, 90")
        try:
            angle_options = [int(x.strip()) for x in angle_input.split(',') if x.strip()]
            st.write(f"✅ Using angles: {angle_options}")
            st.session_state['db_angle_options'] = angle_options
        except ValueError:
            st.error("❌ Invalid angle format. Using default.")
            st.session_state['db_angle_options'] = [0, 45, -45, 90]

        min_plies = st.slider("Min Plies per Section", 2, 12, 3)
        max_plies = st.slider("Max Plies per Section", 3, 16, 6)
        thickness_min = st.number_input("Min Ply Thickness (mm)", 0.05, 0.3, 0.120, step=0.001, format="%.3f")
        thickness_max = st.number_input("Max Ply Thickness (mm)", 0.1, 0.5, 0.191, step=0.001, format="%.3f")
        common_offset = st.radio("Section Offset", ["MID", "TOP", "BOT"], index=0, horizontal=True)

        # Store in session state
        st.session_state['db_params'] = {
            'n_sections': n_sections_db,
            'min_plies': min_plies,
            'max_plies': max_plies,
            'thickness_min': thickness_min,
            'thickness_max': thickness_max,
            'common_offset': common_offset
        }

        st.info("📌 Define mesh, BCs, and loads below. The database will use all current settings.")

# ===============================
# 4. MESH SETTINGS (No Slider)
# ===============================
    st.header("📐 Mesh Settings")
    mesh_cols = st.columns(3)
    with mesh_cols[0]:
        element_size = st.number_input("Element Size", 0.01, 1.0, 0.2, step=0.01, format="%.2f")
    with mesh_cols[1]:
        shape = st.selectbox("Shape", ["QUAD", "TRI"])
    with mesh_cols[2]:
        mesh_type = st.selectbox("Mesh Type", ["FREE", "MAPPED"])

# ===============================
# 5. SECTION → AREA ASSIGNMENT
# ===============================
    st.header("🔗 Section Assignment to Areas")
    n_areas = st.number_input("Number of Areas", 1, 10, 2)

    attributes = []
    for aid in range(n_areas):
        cols = st.columns(2)
        with cols[0]:
            area_id = st.number_input(f"Area {aid+1} ID", 1, 100, 3+aid, key=f"areaid_{aid}")
        with cols[1]:
            sec_id = st.selectbox(f"Assign to Section", list(range(1, n_sections+1)), key=f"secid_{aid}")
        attributes.append({
            "area_id": int(area_id),
            "section_id": int(sec_id)
        })

# ===============================
# 6. BOUNDARY CONDITIONS
# ===============================
    st.header("🔒 Boundary Conditions")
    n_bcs = st.number_input("Number of Boundary Conditions", 1, 10, 1)

    boundary_conditions = []
    valid_entity_types = ["node", "keypoint", "line", "area"]

    for bid in range(n_bcs):
        with st.expander(f"BC {bid+1}", expanded=(bid == 0)):
            cols = st.columns([1, 1, 2])
            
            # --- Type ---
            with cols[0]:
                bc_type = st.selectbox(
                    "Type",
                    ["displacement", "rotation", "symmetry", "fixed"],
                    key=f"bc_type_{bid}"
                )
            
            # --- Entity Type ---
            with cols[1]:
                if bc_type == "symmetry":
                    # Symmetry typically applied on lines/areas
                    entity_type = st.selectbox(
                        "Entity Type",
                        ["line", "area"],
                        key=f"bc_ent_{bid}"
                    )
                else:
                    entity_type = st.selectbox(
                        "Entity Type",
                        valid_entity_types,
                        key=f"bc_ent_{bid}"
                    )
            
            # --- Entity IDs ---
            with cols[2]:
                ids_input = st.text_input(
                    f"{entity_type.capitalize()} IDs (comma-separated)",
                    value="17" if entity_type == "line" else "1",
                    key=f"bc_ids_{bid}"
                )
                try:
                    entity_ids = [int(x.strip()) for x in ids_input.split(',') if x.strip().isdigit()]
                except:
                    entity_ids = [1]
            
            # --- DOF Selection & Values ---
            st.markdown("**Constrained DOFs and Values**")
            dof_cols = st.columns(6)
            dofs = ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"]
            values = {}
            
            if bc_type == "fixed":
                # All DOFs = 0
                values = {dof: 0.0 for dof in dofs}
                st.info("All DOFs fixed to 0.")
            elif bc_type == "symmetry":
                # Common symmetry: UX=0 (YZ-plane), UY=0 (XZ-plane), etc.
                # For simplicity, assume UX=0 for now (you can expand)
                symmetry_plane = st.radio(
                    "Symmetry Plane",
                    ["YZ (UX=0)", "XZ (UY=0)", "XY (UZ=0)"],
                    key=f"sym_plane_{bid}"
                )
                if "YZ" in symmetry_plane:
                    values = {"UX": 0.0}
                elif "XZ" in symmetry_plane:
                    values = {"UY": 0.0}
                else:
                    values = {"UZ": 0.0}
                st.info(f"Symmetry: {', '.join([f'{k}={v}' for k,v in values.items()])}")
            else:
                # Custom displacement or rotation
                for i, dof in enumerate(dofs):
                    default_val = 0.0
                    if dof.startswith("ROT"):
                        unit = "rad"
                        step = 0.01
                    else:
                        unit = "mm"
                        step = 0.01
                    with dof_cols[i]:
                        val = st.number_input(
                            f"{dof} ({unit})",
                            value=default_val,
                            step=step,
                            format="%.4f",
                            key=f"bc_{bid}_{dof}"
                        )
                        if val != 0.0:  # Only store non-zero (or explicitly set)
                            values[dof] = val
            
            # Store BC
            boundary_conditions.append({
                "type": bc_type,
                "entity_type": entity_type,
                "entity_ids": entity_ids,
                "values": values  # e.g., {"UX": 0.1, "UZ": 0.0}
            })
    # st.header("🔒 Boundary Conditions")
    # n_bcs = st.number_input("Number of Fixed Supports", 1, 5, 1)
    #
    # boundary_conditions = []
    # for bid in range(n_bcs):
    #     cols = st.columns(2)
    #     with cols[0]:
    #         line_ids = st.text_input(f"Line IDs (comma-separated)", "17", key=f"bc_lines_{bid}")
    #     with cols[1]:
    #         st.write("DOFs fixed: UX, UY, UZ, ROTX, ROTY, ROTZ")  # always all DOF
    #
    #     try:
    #         ids = [int(x.strip()) for x in line_ids.split(',') if x.strip().isdigit()]
    #     except:
    #         ids = [17]
    #
    #     boundary_conditions.append({
    #         "type": "fixed",
    #         "entity_type": "line",
    #         "entity_ids": ids,
    #         "dof": ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"]
    #     })

# ===============================
# 7. LOADS (Smart: Pressure vs Force/Moment)
# ===============================
    st.header("📌 Loads")

    n_loads = st.number_input("Number of Loads", 1, 10, 1)
    loads = []

    for lid in range(n_loads):
        with st.expander(f"Load {lid+1}"):

            # Step 1: Choose load type
            load_type = st.radio(
                "Load Type",
                ["pressure", "force", "moment"],
                key=f"load_type_{lid}"
            )

            # Step 2: Show only valid entities
            if load_type == "pressure":
                entity_type = st.radio(
                    "Apply on",
                    ["area", "line"],
                    key=f"entity_type_press_{lid}"
                )
                ids_label = f"{entity_type.upper()} IDs"
                ids_help = f"e.g., for {entity_type} 3 and 4: '3, 4'"
            else:
                entity_type = st.radio(
                    "Apply on",
                    ["node", "keypoint"],
                    key=f"entity_type_force_{lid}"
                )
                ids_label = f"{entity_type.upper()} IDs"
                ids_help = f"e.g., for {entity_type}s 101 and 102: '101, 102'"

            # Entity IDs input
            entity_ids_str = st.text_input(
                ids_label,
                "3" if entity_type == "area" else "15" if entity_type == "line" else "201" if entity_type == "node" else "5",
                key=f"load_entities_{lid}",
                help=ids_help
            )

            try:
                entity_ids = [int(x.strip()) for x in entity_ids_str.split(',') if x.strip().isdigit()]
                if not entity_ids:
                    entity_ids = [1]
            except:
                entity_ids = [1]

            # Value input
            if load_type == "pressure":
                value = st.number_input(
                    "Pressure (MPa)",
                    -1000.0, -1.0, -500.0, step=10.0, format="%.1f",
                    key=f"press_val_{lid}"
                )
                loads.append({
                    "type": "pressure",
                    "entity_type": entity_type,
                    "entity_ids": entity_ids,
                    "value": float(value)
                })
            else:
                st.markdown("**Components**")
                cols = st.columns(3)
                fx = fy = fz = mx = my = mz = 0.0

                with cols[0]:
                    if load_type == "force":
                        fx = st.number_input("FX", -10000.0, 10000.0, 0.0, step=10.0, key=f"fx_{lid}")
                        fy = st.number_input("FY", -10000.0, 10000.0, 0.0, step=10.0, key=f"fy_{lid}")
                        fz = st.number_input("FZ", -10000.0, 10000.0, 0.0, step=10.0, key=f"fz_{lid}")
                    elif load_type == "moment":
                        mx = st.number_input("MX", -10000.0, 10000.0, 0.0, step=10.0, key=f"mx_{lid}")
                        my = st.number_input("MY", -10000.0, 10000.0, 0.0, step=10.0, key=f"my_{lid}")
                        mz = st.number_input("MZ", -10000.0, 10000.0, 0.0, step=10.0, key=f"mz_{lid}")

                components = {}
                if load_type == "force":
                    components = {"FX": fx, "FY": fy, "FZ": fz}
                else:
                    components = {"MX": mx, "MY": my, "MZ": mz}

                # Filter out zero components
                components = {k: v for k, v in components.items() if abs(v) > 1e-6}

                loads.append({
                    "type": load_type,
                    "entity_type": entity_type,
                    "entity_ids": entity_ids,
                    "components": components
                })

# ===============================
# 8. OUTPUT OPTIONS
# ===============================
    st.header("💾 Output Options")
    export_disp = st.checkbox("Export Displacement", True)
    export_stress_comp = st.checkbox("Export Stress Components", True)
    export_principal = st.checkbox("Export Principal Stress", True)

# ===============================
# 9. GENERATE config.json
# ===============================
    st.header("✅ Export Configuration")

    if st.button("Generate config.json"):
        config = {
            "geometry": {
                "source": "iges" if geom_source == "IGES" else "box",
                "filename": file_name,
                "path": path if geom_source == "IGES" else "",
                "scale_factor": float(scale_factor)
            },
            "materials": materials,
            "sections": sections,
            "mesh": {
                "element_size": float(element_size),
                "shape": shape,
                "mesh_type": mesh_type
            },
            "attributes": attributes,
            "boundary_conditions": boundary_conditions,
            "loads": loads,
            "output": {
                "export_displacement": export_disp,
                "export_stress_components": export_stress_comp,
                "export_principal_stress": export_principal
            }
        }

        # Save and show
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

        st.success("✅ config.json generated!")
        st.json(config)

        with open('config.json', 'r') as f:
            st.download_button("⬇️ Download config.json", f, "config.json", "application/json")

# # ===============================
# # 10. GENERATE DATABASE (Batch Mode)
# # ===============================
#     st.header("⚡ GENERATE DATABASE (Random Configs)")
#
#     create_db = st.checkbox("Enable Database Generation")
#
#     if create_db:
#         num_configs = st.number_input("Number of Random Configs", 1, 1000, 10)
#         min_plies = st.number_input("Min Plies per Section", 2, 10, 3)
#         max_plies = st.number_input("Max Plies per Section", 3, 12, 6)
#         angle_options = [0, 45, -45, 90, 30, -30, 60]
#
#         if st.button("GENERATE DATABASE"):
#             os.makedirs("configs", exist_ok=True)
#             base_config = {
#                 "geometry": {
#                     "source": "iges" if geom_source == "IGES" else "box",
#                     "filename": file_name,
#                     "path": path if geom_source == "IGES" else "",
#                     "scale_factor": float(scale_factor)
#                 },
#                 "materials": materials,
#                 "mesh": {
#                     "element_size": 0.2,
#                     "shape": "QUAD",
#                     "mesh_type": "FREE"
#                 },
#                 "attributes": [{"area_id": 3, "section_id": 1}, {"area_id": 4, "section_id": 2}],
#                 "boundary_conditions": [{"type": "fixed", "entity_type": "line", "entity_ids": [17], "dof": ["UX","UY","UZ","ROTX","ROTY","ROTZ"]}],
#                 "loads": [{"type": "pressure", "entity_type": "line", "entity_ids": [15], "value": -500.0}],
#                 "output": {
#                     "export_displacement": True,
#                     "export_stress_components": True,
#                     "export_principal_stress": True
#                 }
#             }
#
#             for i in range(num_configs):
#                 n_secs = random.randint(2, 3)
#                 secs = []
#                 attrs = []
#
#                 for sid in range(1, n_secs+1):
#                     n_ply = random.randint(min_plies, max_plies)
#                     plies = []
#                     for _ in range(n_ply):
#                         plies.append({
#                             "thickness": round(random.uniform(0.1, 0.3), 3),
#                             "material_id": random.randint(1, len(materials)),
#                             "angle": random.choice(angle_options),
#                             "integration_points": 3
#                         })
#                     secs.append({
#                         "id": sid,
#                         "type": "SHELL",
#                         "plies": plies,
#                         "offset": "MID"
#                     })
#                     attrs.append({"area_id": 3 if sid==1 else 4, "section_id": sid})
#
#                 config = base_config.copy()
#                 config["sections"] = secs
#                 config["attributes"] = attrs
#                 config["loads"][0]["value"] = round(random.uniform(-1000, -100), 1)
#                 config["mesh"]["element_size"] = round(random.uniform(0.1, 0.5), 2)
#
#                 with open(f"configs/config_{i:03d}.json", 'w') as f:
#                     json.dump(config, f, indent=2)
#
#             st.success(f"✅ Generated {num_configs} configs in /configs/")

    # ===============================
    # FINAL: GENERATE DATASET + CASE FOLDERS
    # ===============================
    st.header("💾 Generate Dataset (Batch Mode)")

    # Input: Number of configurations
    num_configs = st.number_input(
        "Number of Configurations to Generate",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100,
        help="Choose how many random laminate configurations to generate"
    )

    if st.button(f"🛠️ Generate {num_configs} Config Files & Case Folders"):
        if 'db_params' not in st.session_state or 'db_angle_options' not in st.session_state:
            st.error("Please go to 'Generate Database Setup' tab and configure parameters first.")
        else:
            # Save current inputs into session state
            st.session_state['materials'] = materials
            st.session_state['attributes'] = attributes
            st.session_state['boundary_conditions'] = boundary_conditions
            st.session_state['loads'] = loads
            st.session_state['element_size'] = element_size
            st.session_state['shape'] = shape
            st.session_state['mesh_type'] = mesh_type
            st.session_state['export_disp'] = export_disp
            st.session_state['export_stress_comp'] = export_stress_comp
            st.session_state['export_principal'] = export_principal
            st.session_state['file_name'] = file_name
            st.session_state['path'] = path
            st.session_state['scale_factor'] = scale_factor

            # ----------------------------------------
            # STEP 1: Generate config_xxx.json files
            # ----------------------------------------
            generated_count = generate_laminate_database(n_configs=num_configs, config_dir="configs")
            st.success(f"✅ Generated {generated_count:,} config files in `/configs/`")

            # ----------------------------------------
            # STEP 2: Create case_xxx/ folders with APDL
            # ----------------------------------------
            try:
                from utils import generate_all_cases
                with st.spinner("Generating case folders with APDL scripts..."):
                    generate_all_cases(config_dir="configs", cases_dir="cases")
                st.success("✅ All case folders created!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Failed to generate case folders: {e}")
                st.exception(e)

    # if st.button("Build GNN data"):
    #     data = build_gnn_dataset()
    #     # elem = parse_element_table('element_list.txt', 'element_data.txt')
    #     # dlist = parse_dlist('bc_list.txt', 'bc_data.txt')
    #     # plist = parse_plist('pressure_list.txt', 'pressure_data.txt')
    #     if data is not None:
    #         print("🎉 Dataset built successfully!")

