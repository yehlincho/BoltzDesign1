# BoltzDesign1

**BoltzDesign1** is a molecular design tool powered by the Boltz model. This guide provides setup instructions, usage configuration, and evaluation workflow. Example usage can be found in the notebook.

---

## 🛠️ Setup & Installation

Before running BoltzDesign1, configure your environment and install required dependencies.

```bash
conda create -n boltz_env python=3.10 -y
conda activate boltz_env
pip install boltz -U
pip install matplotlib seaborn prody tqdm PyYAML requests
```

### 🔽 Download Boltz Weights and Dependencies

RDKit CCD and Boltz model weights can be downloaded via:

```python
from boltz.main import download
from pathlib import Path

cache = Path("~/.boltz").expanduser()
cache.mkdir(parents=True, exist_ok=True)
download(cache)
```

---

## 📦 Dependencies

Required packages and tools:
- `torch` — PyTorch, the core deep learning framework
- `rdkit` — Toolkit for molecule visualization and manipulation
- `boltz-1` — Core diffusion model
- `LigandMPNN` — Ligand sequence model
- `boltzdesign_utils` — Utility functions for model design pipeline

To set up LigandMPNN and ProteinMPNN:

```bash
cd LigandMPNN
bash get_model_params.sh "./model_params"
```

All code and model weights are provided under the MIT license.

---

## ⚙️ Model Configuration

The Boltz model uses the following default prediction arguments:

```python
predict_args = {
    "recycling_steps": 0,
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "write_confidence_summary": True,
    "write_full_pae": False,
    "write_full_pde": False,
}
```

These parameters can be modified to adjust sampling depth, recycling, and output verbosity.

---

## 🧬 Design Configuration

The molecular design process is controlled using this configuration:

```python
config = {
    'mutation_rate': 1,
    'pre_iteration': 30,
    'soft_iteration': 75,
    'temp_iteration': 45,
    'hard_iteration': 5,
    'semi_greedy_steps': 0,
    'learning_rate_pre': 0.2,
    'learning_rate': 0.1,
    'design_algorithm': '3stages',
    'set_train': True,
    'use_temp': True,
    'disconnect_feats': True,
    'disconnect_pairformer': False,
    'length': 150
}
```

These control learning phases, temperature sampling, and architectural behavior during design.

---

## 🔁 Sequence Redesign

Sequence redesign is supported via:

- **ProteinMPNN** – for redesigning protein–protein interfaces (PPIs)
- **LigandMPNN** – for non-protein biomolecule binding partners

By default, residues at the interface (with CA–CA distance < 5 Å) are fixed, while the rest are redesigned.

---

## ✅ Final Evaluation

Final structure validation is performed using **AlphaFold3 (AF3)**. However, alternative evaluation models like **Chai** or others can be integrated depending on your workflow.
