# WESTPA Easy Setup Pipeline (WESTPA 2.0 + NAMD)

A robust, user-friendly pipeline to automate the setup of [WESTPA 2.0](https://westpa.github.io/westpa/overview.html) weighted ensemble simulations using **NAMD 3.0+** (GPU/CPU). 

This tool abstracts away the complexity of configuring `west.cfg`, `runseg.sh`, and `system.py`, allowing you to focus on your science.

## Key Features

*   **Zero-Config Setup**: Automatically generates all necessary configuration files (`west.cfg`, `runseg.sh`, `md.conf`).
*   **Minimal Adaptive Binning (MAB)**: "Killer feature" support. No need to define bin boundaries manually—the system learns them on the fly.
*   **Multi-Metric Support**: Use **RMSD**, **Distance**, **Radius of Gyration (RoG)**, **SASA**, or **Dihedrals** as your progress coordinate.
*   **HPC Ready**: Auto-generates submission scripts (**Slurm**, **PBS**) tailored to your job.
*   **Robust Analysis**: Includes scripts to recalculate *any* metric (e.g., Free Energy of SASA) from your trajectories, even if you simulated using RMSD.
*   **Secure Inputs**: Safely handles complex MDTraj selection strings (e.g., `'name "CA" and resid 10'`).

## Installation

This pipeline requires a standard scientific Python environment. We recommend using Conda/Mamba.

```bash
# Create environment
conda create -n westpa_pipeline python=3.9 -c conda-forge
conda activate westpa_pipeline

# Install dependencies
conda install westpa mdtraj numpy matplotlib h5py scipy
# OR
pip install -r requirements.txt
```

**Requirements:**
- WESTPA 2.0+
- MDTraj (for coordinate calculation)
- NAMD 2.14 or 3.0+ (installed separately on your system)

## 🛠 Usage

### 1. Basic RMSD Simulation
Set up a simulation using RMSD of the protein backbone as the progress coordinate.

```bash
python setup_westpa.py \
    --pdb common/system.pdb \
    --psf common/system.psf \
    --toppar common/toppar/ \
    --target common/target.pdb \
    --basis-coor eq_files/final.coor \
    --basis-vel eq_files/final.vel \
    --basis-xsc eq_files/final.xsc \
    --output-dir my_sim_01 \
    --pcoord-type rmsd \
    --pcoord-sel "name CA" \
    --walkers 8 --iterations 100
```

### 2. "Auto-Pilot" with MAB (Adaptive Binning)
Don't know your bin boundaries? Use `--adaptive-bins` (MAB) to let WESTPA figure it out.

```bash
python setup_westpa.py ... \
    --pcoord-type distance \
    --pcoord-sel "resid 10" \
    --pcoord-ref "resid 50" \
    --adaptive-bins \
    --max-run-wallclock 24:00:00 \
    --gen-submit slurm --job-name mabs_run
```

### 3. Auxiliary Analysis
After your simulation, you can recalculate free energy surfaces for **new metrics** without re-running the simulation.

```bash
cd my_sim_01/analysis

# Calculate Free Energy of Radius of Gyration
python analyze_alt_metric.py --type rog --sel "protein"

# Calculate Free Energy of Distance
python analyze_alt_metric.py --type distance --sel "resid 30" --sel2 "resid 60"
```

## Directory Structure

The setup script creates a self-contained directory:

```
my_sim_01/
├── west.cfg             # Main configuration (Auto-generated)
├── submit.sh            # Slurm/PBS submission script
├── run.sh               # Local run script
├── init.sh              # Initialization script
├── common_files/        # Copied topology/parameters
├── bstates/             # Basis states
├── westpa_scripts/      # Propagator & Metric scripts
│   ├── runseg.sh
│   ├── calc_pcoord.py   # Dynamic metric calculator
│   └── gen_istate.sh
└── analysis/            # Analysis tools
    ├── analyze_alt_metric.py
    └── ...
```

## Notes
*   **NAMD GPU**: Use `--namd-gpu 0` (or list IDs) to enable GPU acceleration.
*   **Selection Syntax**: Uses [MDTraj selection syntax](https://mdtraj.org/1.9.8.dev0/atom_selection.html).
