# FiberNet Designer: Composite Laminate Simulation and GNN Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)

This project is a comprehensive toolset for designing, simulating, and predicting the behavior of composite laminates. It integrates ANSYS for finite element simulations, generates datasets from simulation results, and uses Graph Neural Networks (GNNs) with physics-informed losses to predict displacements and stresses. The system includes a Streamlit GUI for interactive setup, batch processing for simulations, and model training/inference.

The core focus is on composite materials (e.g., fiber-reinforced laminates) under pressure loads, using principles like the virtual work method for physics-based regularization in the GNN.

## Features

- **Interactive GUI**: Define materials, sections (ply stacks), boundary conditions, loads, and mesh settings using Streamlit.
- **Configuration Generation**: Create multiple laminate configurations (e.g., varying ply angles and thicknesses) and export as JSON.
- **ANSYS Integration**: Generate APDL scripts for ANSYS simulations, run batch jobs, and parse results (displacements, stresses, etc.).
- **Dataset Building**: Parse ANSYS outputs to create PyTorch Geometric datasets for GNN training.
- **GNN Model**: Train a GCN-based model to predict nodal displacements and stresses, with optional physics loss (virtual work principle).
- **Prediction**: Infer on new layups without re-running simulations.
- **Visualization**: Plot training curves, displacement fields, and more using Matplotlib and Plotly.
- **Batch Processing**: Handle multiple cases efficiently for large-scale dataset generation.

## Requirements

- Python 3.8+
- ANSYS (v242 or compatible) installed with MAPDL executable accessible (e.g., `C:\Program Files\ANSYS Inc\v242\ansys\bin\winx64\MAPDL.exe`).
- Libraries: Install via `pip install -r requirements.txt` (create one if needed; key packages include):
  - `streamlit`
  - `torch`
  - `torch-geometric`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `plotly`
  - `scipy`
  - `sklearn`
- Geometry files (e.g., IGES files for import into ANSYS).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-repo/fibernet-designer.git
   cd fibernet-designer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   (Note: Create `requirements.txt` with the listed packages if not present.)

3. Ensure ANSYS is installed and the executable path is correctly referenced in scripts (e.g., `simulate.py`).

4. (Optional) Set up environment variables for ANSYS paths if needed.

## Usage

### 1. Run the Streamlit GUI
Launch the interactive app:
```
streamlit run gui.py
```
- **Setup Tab**: Define materials, sections (plies with angles/thicknesses/materials), geometry, mesh, BCs, loads.
- **Simulation Tab**: Generate configs, build case folders, run ANSYS batch, parse results into dataset.
- **Training Tab**: Train the GNN model on the generated dataset.
- **Model Tab**: Predict displacements/stresses for new layups and visualize results.

### 2. Batch Simulation
- Generate configurations: Run `gen_conf.py` to create multiple JSON configs in `configs/`.
- Generate APDL scripts: Use `conv_gen.py` or integrated in GUI.
- Run simulations: Execute `simulate.py` for batch ANSYS runs.
- Parse results: Run `main.py` or use GUI to build `composite_dataset.pt`.

### 3. Training the GNN
- Load dataset: Use `gnn/load_data.py`.
- Train: Run `gnn/train.py` or via GUI.
- Model: Defined in `gnn/model.py` (GCNConv-based).
- Physics Loss: Implemented in `gnn/physics_loss.py` (virtual work principle for internal/external work balance).

### 4. Prediction
- Load model: Use `gnn/predict.py`.
- Infer: Provide ply angles/thicknesses; outputs displacements, stresses, and coordinates.

### Example Workflow
1. Define base config in GUI and generate database (e.g., 100 configs).
2. Build case folders with `generate_all_cases` in `gui.py`.
3. Run ANSYS batch with `master.py` (generates `master_batch.mac`).
4. Parse results to dataset with `build.py`.
5. Train GNN in `train.py`.
6. Predict new layups in `predict.py` or GUI.

### Key Scripts and Modules
- `gui.py`: Main Streamlit app for end-to-end workflow.
- `master.py`: Generates master ANSYS batch script.
- `simulate.py`: Runs ANSYS on case folders.
- `build.py`: Builds PyTorch Geometric dataset from parsed results.
- `gnn/model.py`: GNN architecture (CompositeGNN).
- `gnn/train.py`: Training loop with resume/stop support.
- `gnn/predict.py`: Inference for new laminates.
- `gnn/physics_loss.py`: Physics-informed loss (strain, stress, virtual work).
- `gnn/load_data.py`: Loads and scales dataset.
- `conv.py`: Converts JSON configs to APDL scripts.
- `gen_conf.py`: Generates variant configs.
- `validate.py`: Validates NPZ dataset.

### Directory Structure
```
.
├── gnn/                # GNN-related modules
│   ├── __init__.py
│   ├── load_data.py    # Dataset loading and scaling
│   ├── main.py         # Entry for parsing
│   ├── model.py        # GNN model definition
│   ├── physics_loss.py # Virtual work loss
│   ├── predict.py      # Inference script
│   └── train.py        # Training script
├── cases/              # Generated case folders (simulation outputs)
├── configs/            # JSON configuration files
├── checkpoints/        # Model checkpoints
├── results/            # Per-case ANSYS outputs (txt files)
├── build.py            # Dataset builder
├── conv.py             # JSON to APDL converter
├── conv_gen.py         # Batch APDL generation
├── gen_conf.py         # Config variant generator
├── gui.py              # Streamlit GUI
├── main.py             # Main parsing entry
├── master.py           # Master batch script generator
├── simulate.py         # ANSYS batch runner
├── validate.py         # Dataset validation
├── README.md
└── requirements.txt    # (Add your dependencies)
```

## Contributing
Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request. Ensure code follows PEP8 and includes tests where applicable.

## License
This project is licensed under the GNU General Public License, version 2 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- ESTIA Research on the support of the development of the project
- Compositadour on the fabrication and testing of composite structures
- Built with PyTorch Geometric for GNNs.
- ANSYS for FEM simulations.
- Streamlit for the GUI.

For issues or questions, open a GitHub issue.
