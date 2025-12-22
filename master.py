# generate_master_batch.py
import json
from pathlib import Path

def select_entities(f, entity_type, entity_ids):
    """Write FLST/FITEM selection and return P51X identifier"""
    entity_type = entity_type.lower()
    n = len(entity_ids)

    if entity_type == 'area':
        f.write(f"FLST,2,{n},5,ORDE,{n}\n")  # 5 = area
        for aid in entity_ids:
            f.write(f"FITEM,2,{aid}\n")
        return "P51X"

    elif entity_type == 'line':
        f.write(f"FLST,2,{n},4,ORDE,{n}\n")  # 4 = line
        for lid in entity_ids:
            f.write(f"FITEM,2,{lid}\n")
        return "P51X"

    elif entity_type == 'node':
        f.write(f"FLST,2,{n},1,ORDE,{n}\n")  # 1 = node
        for nid in entity_ids:
            f.write(f"FITEM,2,{nid}\n")
        return "P51X"

    elif entity_type == 'keypoint':
        f.write(f"FLST,2,{n},3,ORDE,{n}\n")  # 3 = keypoint
        for kid in entity_ids:
            f.write(f"FITEM,2,{kid}\n")
        return "P51X"

    else:
        raise ValueError(f"Unsupported entity_type: {entity_type}")

def generate_master_batch(config_dir="configs", output_file="master_batch.mac"):
    config_dir = Path(config_dir)
    config_files = sorted(config_dir.glob("config_*.json"))
    
    if not config_files:
        raise FileNotFoundError(f"No config files found in {config_dir}")

    with open(output_file, 'w') as f:
        f.write("""! ==================================================================\n""")
        f.write("""! AUTO-GENERATED MASTER BATCH SCRIPT\n""")
        f.write("""! ==================================================================\n\n""")

        # Read first config for shared data
        first_cfg = json.load(open(config_files[0]))
        geom = first_cfg['geometry']
        mesh = first_cfg['mesh']
        materials = first_cfg['materials']
        boundary_conditions = first_cfg['boundary_conditions']
        loads = first_cfg['loads']
        attributes = first_cfg['attributes']  # ← Used once

        # -------------------------------
        # 1. CLEAR + IMPORT GEOMETRY
        # -------------------------------
        f.write("/CLEAR\n")
        f.write(f"/FILNAM, BatchMaster_{len(config_files)}, log\n\n")

        f.write("! --- Import Geometry ---\n")
        f.write(f"*SET, FILENAME, '{geom['filename']}'\n")
        f.write(f"*SET, FILEPATH, '{geom['path']}'\n")
        f.write("/AUX15\n")
        f.write("IOPTN, IGES, SMOOTH\n")
        f.write("IOPTN, MERGE, YES\n")
        f.write("IOPTN, SOLID, YES\n")
        f.write("IGESIN, FILENAME, 'iges', FILEPATH\n")
        f.write("FINISH\n\n")

        # -------------------------------
        # 2. PREP7: Materials, Element, Attributes, Mesh
        # -------------------------------
        f.write("/PREP7\n")
        f.write("ALLSEL\n")
        f.write(f"ARSCALE, ALL, , , {geom['scale_factor']}, {geom['scale_factor']}, {geom['scale_factor']}, , 1, 1\n")
        f.write("KEYOPT, 1,8,1\n")
        f.write("ET, 1, SHELL181\n\n")

        # Materials
        f.write("! --- Materials ---\n")
        for mat in materials:
            mid = mat['id']
            f.write(f"MPTEMP, 1, 0\n")
            f.write(f"MPDATA, EX, {mid}, , {mat['EX']}\n")
            f.write(f"MPDATA, EY, {mid}, , {mat['EY']}\n")
            f.write(f"MPDATA, EZ, {mid}, , {mat['EZ']}\n")
            f.write(f"MPDATA, PRXY, {mid}, , {mat['PRXY']}\n")
            f.write(f"MPDATA, GXY, {mid}, , {mat['GXY']}\n")
            f.write(f"MPDATA, GYZ, {mid}, , {mat['GYZ']}\n")
            f.write(f"MPDATA, GXZ, {mid}, , {mat['GXZ']}\n")
        f.write("\n")

        # === ONCE: Define section IDs and assign to areas ===
#        f.write("! --- Assign Section IDs to Areas (ONCE) ---\n")
#        for attr in attributes:
#            aid = attr['area_id']
#            sec_id = attr['section_id']
#            f.write(f"ASEL, , , , {aid}\n")
#            f.write(f"AATT, 1, , 1, 0, {sec_id}  ! Area {aid} -> Sec {sec_id}\n")
#        f.write("ASEL, ALL\n\n")

        # Mesh
        f.write(f"*SET, ELSIZE, {mesh['element_size']}\n")
        f.write(f"*SET, MSHAPE_VAL, {0 if mesh['shape'] == 'QUAD' else 1}\n")
        f.write(f"*SET, MSHKEY_VAL, {0 if mesh['mesh_type'] == 'FREE' else 1}\n")
        f.write("AESIZE, ALL, ELSIZE\n")
        f.write("MSHAPE, MSHAPE_VAL, 2D\n")
        f.write("MSHKEY, MSHKEY_VAL\n")
        f.write("AMESH, ALL\n\n")

        # Cleanup
        f.write("CMDELE, ALL\n")
        f.write("FINISH\n\n")

        # -------------------------------
        # 3. LOOP OVER CONFIGS: ONLY CHANGE SECTIONS
        # -------------------------------
        f.write("! ==================================================================\n")
        f.write("! STARTING SIMULATION LOOP\n")
        f.write("! ONLY REDEFINE SECTIONS, NOT ATTRIBUTES OR MESH\n")
        f.write("! ==================================================================\n\n")

        for i, config_path in enumerate(config_files):
            cfg = json.load(open(config_path))
            sim_id = i + 1
            case_name = f"case_{sim_id:03d}"
            case_folder = Path("cases") / case_name
            result_dir = (case_folder / "results").resolve()  # ← Full path
            # result_dir = f"cases/{case_name}/results"
            result_dir.mkdir(parents=True, exist_ok=True)

            f.write(f"! =========================================\n")
            f.write(f"! SIMULATION {sim_id}: {case_name}\n")
            f.write(f"! =========================================\n\n")


            # Create results dir
            f.write(f"/SYS, mkdir -p \"{result_dir}\" \n\n")

            # Solution control
            f.write("/SOLU\n")
            f.write("ANTYPE, STATIC\n\n")

            # === Boundary Conditions ===
            f.write("! --- Boundary Conditions ---\n")
            for bc in boundary_conditions:
                if bc['type'] != 'fixed':
                    continue  # Extend later for 'pinned', etc.
            
                entity_type = bc['entity_type'].lower()
                entity_ids = bc['entity_ids']
            
                p51x = select_entities(f, entity_type, entity_ids)
            
                if entity_type == 'area':
                    f.write(f"DA, {p51x}, ALL, 0   ! Fixed area\n")
                elif entity_type == 'line':
                    f.write(f"DL, {p51x}, , ALL, 0   ! Fixed line\n")
                elif entity_type == 'node':
                    f.write(f"D, {p51x}, ALL, 0   ! Fixed node\n")
                elif entity_type == 'keypoint':
                    f.write(f"DK, {p51x}, ALL, 0   ! Fixed keypoint\n")
            
            f.write("\n")
            
            # === Loads ===
            f.write("! --- Loads ---\n")
            for load in loads:
                if load['type'] != 'pressure':
                    continue  # Later: add 'force', 'moment'
            
                value = load['value']
                entity_type = load['entity_type'].lower()
                entity_ids = load['entity_ids']
            
                p51x = select_entities(f, load['entity_type'], entity_ids)
            
                if entity_type == 'area':
                    f.write(f"SFA, {p51x}, 1, PRES, {value}   ! Pressure on area\n")
                elif entity_type == 'line':
                    f.write(f"SFL, {p51x}, PRES, {value}, ,   ! Line pressure\n")
                elif entity_type == 'node':
                    # Nodal pressure = concentrated force (convert if needed)
                    f.write(f"F, {p51x}, FZ, {value}   ! Nodal force in Z (example)\n")
                    print(f"Warning: Nodal 'pressure' interpreted as FZ = {value}")
                elif entity_type == 'keypoint':
                    f.write(f"FK, {p51x}, FZ, {value}   ! Keypoint force in Z\n")
                    print(f"Warning: Keypoint 'pressure' interpreted as FZ = {value}")
            
            f.write("\n")

            # ===============================
            # REDFINE SECTIONS ONLY
            # ===============================
            f.write("/PREP7\n")
            sections = cfg['sections']

            for sec in sections:
                sid = sec['id']
                offset = sec.get('offset', 'MID').upper()
                plies = sec['plies']

                f.write(f"! --- Redefine Section {sid} ---\n")
                f.write(f"SECT, {sid}, SHELL\n")
                for ply in plies:
                    t = ply['thickness']
                    mat_id = ply['material_id']
                    angle = ply['angle']
                    ipoints = ply['integration_points']
                    f.write(f"SECData, {t}, {mat_id}, {angle}, {ipoints}\n")
                f.write(f"SECOFFSET, {offset}\n")
                f.write(f"SECCONTROL, 0,0,0,0,1,1,1\n")

            # Re-apply AATT
            f.write("! --- Re-link areas ---\n")
            for attr in cfg.get('attributes', []):
                aid = attr['area_id']
                sec_id = attr['section_id']
                f.write(f"ASEL, , , , {aid}\n")
                f.write(f"AATT, 1, , 1, 0, {sec_id}\n")
            f.write("ASEL, ALL\n\n")

            f.write("FINISH\n")
            f.write("/SOLU\n")
            f.write("SOLVE\n")
            f.write("FINISH\n\n")

            # ===============================
            # POST-PROCESSING: Export Results
            # ===============================
            f.write("/POST1\n")
            f.write("SET, LAST\n\n")

            # Lists
            f.write(f"/OUTPUT, node_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("NLIST,ALL, , , ,NODE,NODE,NODE\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, strain_node_prin_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRNSOL, EPEL, PRIN\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, strain_elem_prin_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRESOL, EPEL, PRIN\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, strain_node_comp_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRNSOL, EPEL, COMP\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, strain_elem_comp_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRESOL, EPEL, COMP\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, stress_elem_prin_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRESOL, S, PRIN\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, stress_node_prin_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRNSOL, S, PRIN\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, stress_elem_comp_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRESOL, S, COMP\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, stress_node_comp_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRNSOL, S, COMP\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, disp_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("PRNSOL, U, COMP\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, element_list, txt, '{result_dir}',,0\n")
            f.write("ALLSEL, ALL\n")
            f.write("ELIST, ALL\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, bc_list, txt, '{result_dir}',,0\n")
            f.write("DLIST, ALL\n")
            f.write("/OUTPUT\n\n")

            f.write(f"/OUTPUT, pressure_list, txt, '{result_dir}',,0\n")
            f.write("PLIST, ALL\n")
            f.write("/OUTPUT\n\n")

            f.write("FINISH\n\n")

        f.write("!!! BATCH SIMULATION COMPLETE !!!\n")
        print(f"✅ Generated: {output_file}")

# generate_master_batch()
