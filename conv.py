from pathlib import Path
import json
import os

def generate_apdl_from_json(config_path, case_dir, config_file_name='config.json'):
    """
    Reads config.json and generates:
      - params.apdl
      - sections.apdl
      - mesh_attr.apdl
      - bc.apdl
    Saves config.json into case folder.
    """
    # Convert to Path
    config_path = Path(config_path)
    case_dir = Path(case_dir)
    apdl_dir = case_dir / "apdl_generated"
    apdl_dir.mkdir(parents=True, exist_ok=True)

    # Read input config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {config_path}: {e}")
        raise

    # Copy config.json into case folder
    dest_config = case_dir / config_file_name
    with open(dest_config, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    print(f"✅ Copied config to {dest_config}")

    # APDL file paths
    params_file = apdl_dir / 'params.apdl'
    sections_file = apdl_dir / 'sections.apdl'
    bc_file = apdl_dir / 'bc.apdl'
    mesh_attr_file = apdl_dir / 'mesh_attr.apdl'

    # ===============================
    # 1. params.apdl
    # ===============================
    with open(params_file, 'w', encoding='utf-8') as f:
        geom = cfg.get('geometry', {})
        materials = cfg.get('materials', [])
        mesh = cfg.get('mesh', {})

        f.write("/PREP7\n")
        f.write(f"*SET, GEOM_SCALE, {geom.get('scale_factor', 1.0)}\n")
        f.write(f"*SET, FILENAME, '{geom.get('filename', 'plate')}'\n")
        f.write(f"*SET, FILEPATH, '{geom.get('path', '.')}'\n\n")

        f.write("! --- Materials ---\n")
        for mat in materials:
            mid = mat['id']
            mname = mat['name']
            f.write(f"! Material {mid}: {mname}\n")
            f.write(f"MPTEMP, 1, 0\n")
            f.write(f"MPDATA, EX, {mid}, , {mat['EX']}\n")
            f.write(f"MPDATA, EY, {mid}, , {mat['EY']}\n")
            f.write(f"MPDATA, EZ, {mid}, , {mat['EZ']}\n")
            f.write(f"MPDATA, PRXY, {mid}, , {mat['PRXY']}\n")
            f.write(f"MPDATA, PRYZ, {mid}, , {mat['PRYZ']}\n")
            f.write(f"MPDATA, PRXZ, {mid}, , {mat['PRXZ']}\n")
            f.write(f"MPDATA, GXY, {mid}, , {mat['GXY']}\n")
            f.write(f"MPDATA, GYZ, {mid}, , {mat['GYZ']}\n")
            f.write(f"MPDATA, GXZ, {mid}, , {mat['GXZ']}\n\n")

        f.write("! --- Mesh Settings ---\n")
        f.write(f"*SET, ELSIZE, {mesh.get('element_size', 0.1)}\n")
        f.write(f"*SET, MSHAPE_VAL, {0 if mesh.get('shape', 'QUAD') == 'QUAD' else 1}\n")
        f.write(f"*SET, MSHKEY_VAL, {0 if mesh.get('mesh_type', 'FREE') == 'FREE' else 1}\n")

    # ===============================
    # 2. mesh_attr.apdl
    # ===============================
    with open(mesh_attr_file, 'w', encoding='utf-8') as f:
        f.write("/PREP7\n")
        attributes = cfg.get('attributes', [])
        for assign in attributes:
            aid = assign['area_id']
            sec_id = assign['section_id']
            mat_id = assign.get('material_id', 1)
            elem_type = assign.get('element_type', 1)
            coord_sys = assign.get('coord_sys', 0)

            f.write(f"! Area {aid} → Sec {sec_id}\n")
            f.write(f"CM, A{aid}_TEMP, AREA\n")
            f.write(f"ASEL, , , , {aid}\n")
            f.write(f"CM, A{aid}_SEL, AREA\n")
            f.write(f"AATT, {mat_id}, , {elem_type}, {coord_sys}, {sec_id}\n")
            f.write(f"CMSEL, S, A{aid}_TEMP\n")
            f.write(f"CMDELE, A{aid}_TEMP\n")
            f.write(f"CMDELE, A{aid}_SEL\n\n")

    # ===============================
    # 3. bc.apdl
    # ===============================
    with open(bc_file, 'w', encoding='utf-8') as f:
        f.write("! ==================================================================\n")
        f.write("! AUTO-GENERATED: BOUNDARY CONDITIONS AND LOADS (FOR /SOLU ONLY)\n")
        f.write("! ==================================================================\n\n")

        bcs = cfg.get('boundary_conditions', [])
        for bc in bcs:
            entity_type = bc['entity_type']
            entity_ids = bc['entity_ids']
            values = bc.get('values', {})

            # Skip if no DOFs are constrained
            if not values:
                print(f"⚠️ Skipping BC: no 'values' defined for entity {entity_ids}")
                continue

            # Map entity type to ANSYS FLST type code
            if entity_type == "node":
                flst_type = 1  # NODE
            elif entity_type == "keypoint":
                flst_type = 3  # KP
            elif entity_type == "line":
                flst_type = 4  # LINE
            elif entity_type == "area":
                flst_type = 5  # AREA
            else:
                continue

            # Write FLST/FITEM block
            f.write(f"FLST,2,{len(entity_ids)},{flst_type},ORDE,{len(entity_ids)}\n")
            for eid in entity_ids:
                f.write(f"FITEM,2,{eid}\n")

            # Apply each DOF constraint (UX, UY, ..., ROTZ)
            for dof, val in values.items():
                # ANSYS accepts: UX, UY, UZ, ROTX, ROTY, ROTZ
                f.write(f"DL, P51X, , {dof}, {val}   ! Prescribed {dof} = {val}\n")
            f.write("\n")
        # bcs = cfg.get('boundary_conditions', [])
        # for bc in bcs:
        #     if bc['type'] == 'fixed' and bc['entity_type'] == 'line':
        #         lids = bc['entity_ids']
        #         f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
        #         for lid in lids:
        #             f.write(f"FITEM,2,{lid}\n")
        #         f.write(f"DL, P51X, , ALL, 0   ! Fixed support\n")
        # f.write("\n")

        loads = cfg.get('loads', [])
        for load in loads:
            if load['type'] == 'pressure' and load['entity_type'] == 'area':
                aids = load['entity_ids']
                f.write(f"FLST,2,{len(aids)},5,ORDE,{len(aids)}\n")
                for aid in aids:
                    f.write(f"FITEM,2,{aid}\n")
                f.write(f"SFA, P51X, 1, PRES, {load['value']}   ! Pressure\n")

            elif load['type'] == 'pressure' and load['entity_type'] == 'line':
                lids = load['entity_ids']
                f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
                for lid in lids:
                    f.write(f"FITEM,2,{lid}\n")
                f.write(f"SFL, P51X, PRES, {load['value']}   ! Line pressure\n")

            elif load['type'] == 'force' and load['entity_type'] == 'node':
                nids = load['entity_ids']
                f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")
                for nid in nids:
                    f.write(f"FITEM,2,{nid}\n")
                for dof, val in load['components'].items():
                    if dof in ['FX','FY','FZ']:
                        f.write(f"F, P51X, {dof}, {val}\n")

            elif load['type'] == 'moment' and load['entity_type'] == 'node':
                nids = load['entity_ids']
                f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")
                for nid in nids:
                    f.write(f"FITEM,2,{nid}\n")
                for dof, val in load['components'].items():
                    if dof in ['MX','MY','MZ']:
                        rot_dof = dof.replace('M', 'ROT')
                        f.write(f"F, P51X, {rot_dof}, {val}\n")

    # ===============================
    # 4. sections.apdl
    # ===============================
    with open(sections_file, 'w', encoding='utf-8') as f:
        f.write("/PREP7\n")
        sections = cfg.get('sections', [])
        for sec in sections:
            sid = sec['id']
            offset = sec.get('offset', 'MID').upper()
            plies = sec['plies']

            f.write(f"SECT, {sid}, SHELL\n")
            for ply in plies:
                t = ply['thickness']
                mat_id = ply['material_id']
                angle = ply['angle']
                ipoints = ply['integration_points']
                f.write(f"SECData, {t}, {mat_id}, {angle}, {ipoints}\n")
            f.write(f"SECOFFSET, {offset}\n")
            #f.write(f"SECCONTROL, 0,0,0,0,1,1,1\n")
        f.write("FINISH\n")

    print(f"✅ Generated APDL files in: {apdl_dir}")

# generate_apdl_from_json("configs/", "cases", "config_0000.json")





# def generate_apdl_from_json(config_path, config_file='config.json', case_dir):
#     """
#     Reads config.json and generates:
#       - params.apdl  : materials, BCs, loads, mesh
#       - sections.apdl: section definitions with variable plies
#     """
#     case_dir = Path(case_dir)
#     apdl_dir = case_dir / "apdl_generated"
#     apdl_dir.mkdir(parents=True, exist_ok=True)
#     # Copy config.json to case folder
#     with open(config_path, 'r') as f:
#         cfg = json.load(f)
#     with open(case_dir / config_file, 'w') as f:
#         json.dump(cfg, f, indent=2)
#     with open(config_file, 'r') as f:
#         cfg = json.load(f)
#
#     # Ensure output directory exists
#     os.makedirs(apdl_dir, exist_ok=True)
#     params_file = apdl_dir / 'params.apdl'
#     sections_file = apdl_dir / 'sections.apdl'
#     bc_file = apdl_dir / 'bc.apdl'
#     mesh_attr = apdl_dir / 'mesh_attr.apdl'
#
#     # -------------------------------
#     # Generate params.apdl
#     # -------------------------------
#     with open(params_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: MATERIALS, LOADS, BCS, MESH\n")
#         f.write("! Source: config.json\n")
#         f.write("! ==================================================================\n\n")
#
#         geom = cfg.get('geometry', {})
#         f.write("/PREP7\n")  # ← Add this
#         f.write(f"! --- Geometry ---\n")
#         f.write(f"*SET, GEOM_SCALE, {geom.get('scale_factor', 1.0)}\n")
#         f.write(f"*SET, FILENAME, '{geom.get('filename', 'plate')}'\n")
#         f.write(f"*SET, FILEPATH, '{geom.get('path', '.')}'\n\n")
#
#         f.write(f"! --- Materials ---\n")
#         materials = cfg.get('materials', [])
#         for mat in materials:
#             mid = mat['id']
#             mname = mat['name']
#             f.write(f"! --- Material {mid}: {mname} ---\n")
#             f.write(f"MPTEMP, 1, 0\n")
#             f.write(f"MPDATA, EX, {mid}, , {mat['EX']}\n")
#             f.write(f"MPDATA, EY, {mid}, , {mat['EY']}\n")
#             f.write(f"MPDATA, EZ, {mid}, , {mat['EZ']}\n")
#             f.write(f"MPDATA, PRXY, {mid}, , {mat['PRXY']}\n")
#             f.write(f"MPDATA, PRYZ, {mid}, , {mat['PRYZ']}\n")
#             f.write(f"MPDATA, PRXZ, {mid}, , {mat['PRXZ']}\n")
#             f.write(f"MPDATA, GXY, {mid}, , {mat['GXY']}\n")
#             f.write(f"MPDATA, GYZ, {mid}, , {mat['GYZ']}\n")
#             f.write(f"MPDATA, GXZ, {mid}, , {mat['GXZ']}\n")
#             f.write(f"\n")
#         f.write("\n")
#
#         f.write(f"! --- Mesh Settings ---\n")
#         mesh = cfg.get('mesh', {})
#         f.write(f"*SET, ELSIZE, {mesh.get('element_size', 0.1)}\n")
#         f.write(f"*SET, MSHAPE_VAL, {0 if mesh.get('shape', 'QUAD') == 'QUAD' else 1}\n")
#         f.write(f"*SET, MSHKEY_VAL, {0 if mesh.get('type', 'FREE') == 'FREE' else 1}\n\n")
#
#     # -------------------------------
#     # Generate mesh_attr.apdl
#     # -------------------------------
#     with open(mesh_attr, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: Area to Section Assignments\n")
#         f.write("! ==================================================================\n\n")
#
#         attributes = cfg.get('attributes', [])
#
#         for assign in attributes:
#             aid = assign['area_id']
#             sec_id = assign['section_id']
#             mat_id = assign.get('material_id', 1)
#             elem_type = assign.get('element_type', 1)
#             coord_sys = assign.get('coord_sys', 0)
#
#             # Create component for area
#             f.write(f"! --- Assign Section {sec_id} to Area {aid} ---\n")
#             f.write(f"CM, A{aid}_TEMP, AREA\n")
#             f.write(f"ASEL, , , , {aid}\n")
#             f.write(f"CM, A{aid}_SEL, AREA\n")
#             f.write(f"AATT, {mat_id}, , {elem_type}, {coord_sys}, {sec_id}  ! Area {aid} Sec {sec_id}\n")
#             f.write(f"CMSEL, S, A{aid}_TEMP\n")
#             f.write(f"CMDELE, A{aid}_TEMP\n")
#             f.write(f"CMDELE, A{aid}_SEL\n\n")
#
#     # -------------------------------
#     # Generate bc.apdl
#     # -------------------------------
#     with open(bc_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: BOUNDARY CONDITIONS AND LOADS\n")
#         f.write("! ==================================================================\n\n")
#
#         bcs = cfg.get('boundary_conditions', [])
#         for bc in bcs:
#             if bc['type'] == 'fixed' and bc['entity_type'] == 'line':
#                 lids = bc['entity_ids']
#                 f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
#                 for lid in lids:
#                     f.write(f"FITEM,2,{lid}\n")
#                 f.write(f"DL, P51X, , ALL, 0   ! Fixed support\n")
#             # if bc['type'] == 'dx' and bc['entity_type'] == 'line':
#             #     lids = bc['entity_ids']
#             #     f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
#             #     for lid in lids:
#             #         f.write(f"FITEM,2,{lid}\n")
#             #     f.write(f"DL, P51X, , DX,    ! Fixed support\n")
#         f.write("\n")
#
#         f.write(f"! --- Loads ---\n")
#         loads = cfg.get('loads', [])
#         for load in loads:
#             if load['type'] == 'pressure' and load['entity_type'] == 'area':
#                 aids = load['entity_ids']
#                 f.write(f"FLST,2,{len(aids)},5,ORDE,{len(aids)}\n")
#                 for aid in aids:
#                     f.write(f"FITEM,2,{aid}\n")
#                 f.write(f"SFA, P51X, 1, PRES, {load['value']}   ! Pressure load\n")
#
#             if load['type'] == 'pressure' and load['entity_type'] == 'line':
#                 aids = load['entity_ids']
#                 f.write(f"FLST,2,{len(aids)},5,ORDE,{len(aids)}\n")
#                 for aid in aids:
#                     f.write(f"FITEM,2,{aid}\n")
#                 f.write(f"SFL, P51X, PRES, {load['value']}   ! Pressure load\n")
#             # ADD FORCE ON NODE OR KEYPOINT
#             elif load['type'] == 'force' and load['entity_type'] == 'node':
#                 nids = load['entity_ids']
#                 f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")  # NODESEL
#                 for nid in nids:
#                     f.write(f"FITEM,2,{nid}\n")
#                 components = load['components']
#                 for dof, value in components.items():
#                     if dof in ['FX', 'FY', 'FZ'] and abs(value) > 1e-6:
#                         f.write(f"F, P51X, {dof}, {value}   ! Force on nodes\n")
#
#             elif load['type'] == 'force' and load['entity_type'] == 'keypoint':
#                 kps = load['entity_ids']
#                 f.write(f"FLST,2,{len(kps)},3,ORDE,{len(kps)}\n")  # KPSEL
#                 for kp in kps:
#                     f.write(f"FITEM,2,{kp}\n")
#                 components = load['components']
#                 for dof, value in components.items():
#                     if dof in ['FX', 'FY', 'FZ'] and abs(value) > 1e-6:
#                         f.write(f"FK, P51X, {dof}, {value}   ! Force on keypoints\n")
#
#             elif load['type'] == 'moment' and load['entity_type'] == 'node':
#                 nids = load['entity_ids']
#                 f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")
#                 for nid in nids:
#                     f.write(f"FITEM,2,{nid}\n")
#                 components = load['components']
#                 for dof, value in components.items():
#                     if dof in ['MX', 'MY', 'MZ'] and abs(value) > 1e-6:
#                         moment_dof = dof.replace('M', 'ROT')  # ANSYS uses ROTX, etc.
#                         f.write(f"F, P51X, {moment_dof}, {value}   ! Moment on nodes\n")
#
#             elif load['type'] == 'moment' and load['entity_type'] == 'keypoint':
#                 kps = load['entity_ids']
#                 f.write(f"FLST,2,{len(kps)},3,ORDE,{len(kps)}\n")
#                 for kp in kps:
#                     f.write(f"FITEM,2,{kp}\n")
#                 components = load['components']
#                 for dof, value in components.items():
#                     if dof in ['MX', 'MY', 'MZ'] and abs(value) > 1e-6:
#                         moment_dof = dof.replace('M', 'ROT')
#                         f.write(f"FK, P51X, {moment_dof}, {value}   ! Moment on keypoints\n")
#         f.write("\n")
#
#     # -------------------------------
#     # Generate sections.apdl
#     # -------------------------------
#     with open(sections_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: COMPOSITE LAMINATE SECTIONS\n")
#         f.write("! Source: config.json\n")
#         f.write("! ==================================================================\n\n")
#
#         sections = cfg.get('sections', [])
#         f.write("/PREP7\n")  # ← Add this
#         for sec in sections:
#             sid = sec['id']
#             offset = sec.get('offset', 'MID')
#             plies = sec['plies']
#
#             f.write(f"! --- Section {sid}: {len(plies)}-ply laminate ---\n")
#             f.write(f"SECT, {sid}, SHELL\n")
#             for ply in plies:
#                 t = ply['thickness']
#                 mat_id = ply['material_id']
#                 angle = ply['angle']
#                 ipoints = ply['integration_points']
#                 f.write(f"SECData, {t}, {mat_id}, {angle}, {ipoints}  ! t={t}, theta={angle} deg\n")
#             f.write(f"SECOFFSET, {offset}\n")
#             f.write(f"SECCONTROL, 0,0,0,0,1,1,1\n\n")
#         f.write("FINISH\n")
#
#     print(f"Generated:")
#     print(f"   - {params_file}")
#     print(f"   - {sections_file}")
#     print(f"   - {mesh_attr}")
#     print(f"   - {bc_file}")
#     print(f"Now run ANSYS with main_script.mac")

# generate_apdl_from_json("configs/config_0001.json")

# import json
# import os
#
# def generate_apdl_from_config(config_path, output_dir):
#     """
#     Generate full APDL set for one config.json.
#
#     Args:
#         config_path (str): Path to config JSON file
#         output_dir (str): Output folder (e.g., 'apdl_generated/config_000')
#     """
#     with open(config_path, 'r') as f:
#         cfg = json.load(f)
#
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
#
#     # File paths
#     params_file = os.path.join(output_dir, 'params.apdl')
#     sections_file = os.path.join(output_dir, 'sections.apdl')
#     mesh_attr_file = os.path.join(output_dir, 'mesh_attr.apdl')
#     bc_file = os.path.join(output_dir, 'bc.apdl')
#     main_script = os.path.join(output_dir, 'main_script.mac')
#
#     # ===============================
#     # 1. params.apdl: Materials, Mesh Settings
#     # ===============================
#     with open(params_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: MATERIALS, MESH SETTINGS\n")
#         f.write("! Source: config.json\n")
#         f.write("! ==================================================================\n\n")
#
#         geom = cfg.get('geometry', {})
#         f.write("/PREP7\n")
#         f.write(f"*SET, GEOM_SCALE, {geom.get('scale_factor', 1.0)}\n")
#         f.write(f"*SET, FILENAME, '{geom.get('filename', 'plate')}'\n")
#         f.write(f"*SET, FILEPATH, '{geom.get('path', '.')}'\n\n")
#
#         f.write(f"! --- Materials ---\n")
#         materials = cfg.get('materials', [])
#         for mat in materials:
#             mid = mat['id']
#             mname = mat['name']
#             f.write(f"! --- Material {mid}: {mname} ---\n")
#             f.write(f"MPTEMP, 1, 0\n")
#             f.write(f"MPDATA, EX, {mid}, , {mat['EX']}\n")
#             f.write(f"MPDATA, EY, {mid}, , {mat['EY']}\n")
#             f.write(f"MPDATA, EZ, {mid}, , {mat['EZ']}\n")
#             f.write(f"MPDATA, PRXY, {mid}, , {mat['PRXY']}\n")
#             f.write(f"MPDATA, PRYZ, {mid}, , {mat['PRYZ']}\n")
#             f.write(f"MPDATA, PRXZ, {mid}, , {mat['PRXZ']}\n")
#             f.write(f"MPDATA, GXY, {mid}, , {mat['GXY']}\n")
#             f.write(f"MPDATA, GYZ, {mid}, , {mat['GYZ']}\n")
#             f.write(f"MPDATA, GXZ, {mid}, , {mat['GXZ']}\n\n")
#
#         f.write(f"! --- Mesh Settings ---\n")
#         mesh = cfg.get('mesh', {})
#         f.write(f"*SET, ELSIZE, {mesh.get('element_size', 0.1)}\n")
#         f.write(f"*SET, MSHAPE_VAL, {0 if mesh.get('shape', 'QUAD') == 'QUAD' else 1}\n")
#         f.write(f"*SET, MSHKEY_VAL, {0 if mesh.get('mesh_type', 'FREE') == 'FREE' else 1}\n\n")
#
#     # ===============================
#     # 2. mesh_attr.apdl: Section Assignments
#     # ===============================
#     with open(mesh_attr_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: Area to Section Assignments\n")
#         f.write("! ==================================================================\n\n")
#
#         attributes = cfg.get('attributes', [])
#         for assign in attributes:
#             aid = assign['area_id']
#             sec_id = assign['section_id']
#             mat_id = assign.get('material_id', 1)
#             elem_type = assign.get('element_type', 1)
#             coord_sys = assign.get('coord_sys', 0)
#
#             f.write(f"! --- Assign Section {sec_id} to Area {aid} ---\n")
#             f.write(f"CM, A{aid}_TEMP, AREA\n")
#             f.write(f"ASEL, , , , {aid}\n")
#             f.write(f"CM, A{aid}_SEL, AREA\n")
#             f.write(f"AATT, {mat_id}, , {elem_type}, {coord_sys}, {sec_id}\n")
#             f.write(f"CMSEL, S, A{aid}_TEMP\n")
#             f.write(f"CMDELE, A{aid}_TEMP\n")
#             f.write(f"CMDELE, A{aid}_SEL\n\n")
#
#     # ===============================
#     # 3. bc.apdl: Boundary Conditions & Loads
#     # ===============================
#     with open(bc_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: BOUNDARY CONDITIONS AND LOADS\n")
#         f.write("! ==================================================================\n\n")
#
#         bcs = cfg.get('boundary_conditions', [])
#         for bc in bcs:
#             if bc['type'] == 'fixed' and bc['entity_type'] == 'line':
#                 lids = bc['entity_ids']
#                 f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
#                 for lid in lids:
#                     f.write(f"FITEM,2,{lid}\n")
#                 f.write(f"DL, P51X, , ALL, 0   ! Fixed support\n")
#         f.write("\n")
#
#         loads = cfg.get('loads', [])
#         for load in loads:
#             if load['type'] == 'pressure' and load['entity_type'] == 'area':
#                 aids = load['entity_ids']
#                 f.write(f"FLST,2,{len(aids)},5,ORDE,{len(aids)}\n")
#                 for aid in aids:
#                     f.write(f"FITEM,2,{aid}\n")
#                 f.write(f"SFA, P51X, 1, PRES, {load['value']}   ! Pressure load\n")
#
#             elif load['type'] == 'pressure' and load['entity_type'] == 'line':
#                 lids = load['entity_ids']
#                 f.write(f"FLST,2,{len(lids)},4,ORDE,{len(lids)}\n")
#                 for lid in lids:
#                     f.write(f"FITEM,2,{lid}\n")
#                 f.write(f"SFL, P51X, PRES, {load['value']}   ! Line pressure\n")
#
#             elif load['type'] == 'force' and load['entity_type'] == 'node':
#                 nids = load['entity_ids']
#                 f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")
#                 for nid in nids:
#                     f.write(f"FITEM,2,{nid}\n")
#                 for dof, val in load['components'].items():
#                     if dof in ['FX','FY','FZ']:
#                         f.write(f"F, P51X, {dof}, {val}\n")
#
#             elif load['type'] == 'moment' and load['entity_type'] == 'node':
#                 nids = load['entity_ids']
#                 f.write(f"FLST,2,{len(nids)},1,ORDE,{len(nids)}\n")
#                 for nid in nids:
#                     f.write(f"FITEM,2,{nid}\n")
#                 for dof, val in load['components'].items():
#                     if dof in ['MX','MY','MZ']:
#                         rot_dof = dof.replace('M', 'ROT')
#                         f.write(f"F, P51X, {rot_dof}, {val}\n")
#
#     # ===============================
#     # 4. sections.apdl: Composite Sections
#     # ===============================
#     with open(sections_file, 'w') as f:
#         f.write("! ==================================================================\n")
#         f.write("! AUTO-GENERATED: COMPOSITE LAMINATE SECTIONS\n")
#         f.write("! ==================================================================\n\n")
#         f.write("/PREP7\n")
#
#         sections = cfg.get('sections', [])
#         for sec in sections:
#             sid = sec['id']
#             offset = sec.get('offset', 'MID').upper()
#             plies = sec['plies']
#
#             f.write(f"! --- Section {sid} ---\n")
#             f.write(f"SECT, {sid}, SHELL\n")
#             for ply in plies:
#                 t = ply['thickness']
#                 mat_id = ply['material_id']
#                 angle = ply['angle']
#                 ipoints = ply['integration_points']
#                 f.write(f"SECData, {t}, {mat_id}, {angle}, {ipoints}\n")
#             f.write(f"SECOFFSET, {offset}\n")
#             f.write(f"SECCONTROL, 0,0,0,0,1,1,1\n\n")
#         f.write("FINISH\n")
#
#     # ===============================
#     # 5. main_script.mac: Master Script
#     # ===============================
#     with open(main_script, 'w') as f:
#         basename = os.path.splitext(os.path.basename(config_path))[0]
#         f.write(f"! ==================================================================\n")
#         f.write(f"! MASTER SCRIPT FOR {basename}\n")
#         f.write(f"! Generated by FiberNet GUI\n")
#         f.write(f"! ==================================================================\n\n")
#
#         f.write("/BATCH, 1\n")
#         f.write("/INPUT, '%FILEPATH%' + '%FILENAME%.iges', IGES\n")
#         f.write(f"/INPUT, params.apdl\n")
#         f.write(f"/INPUT, sections.apdl\n")
#         f.write(f"/INPUT, mesh_attr.apdl\n")
#         f.write(f"AMESH, ALL\n")
#         f.write(f"/INPUT, bc.apdl\n")
#         f.write(f"/SOLU\n")
#         f.write(f"SOLVE\n")
#         f.write(f"FINISH\n")
#         f.write(f"! Export results\n")
#         f.write(f"/POST1\n")
#         f.write(f"SET, LAST\n")
#         f.write(f"VWRITE, 'Results saved for {basename}'\n")
#         f.write(f"(A)\n")
#
#     print(f"✅ Generated APDL files in: {output_dir}")
#
#
# # ===========================
# # Run the function
# # ===========================
# if __name__ == "__main__":
#     generate_apdl_from_json('configs/config_0000.json')
