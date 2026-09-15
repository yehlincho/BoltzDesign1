# BoltzDesign1 🧬

**BoltzDesign1** is a molecular design tool powered by the Boltz model for designing protein-protein interactions and biomolecular complexes.

> 📄 **Paper**: [BoltzDesign1: AI-Powered Molecular Design](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1)  
> 🚀 **Colab**: https://colab.research.google.com/github/yehlincho/BoltzDesign1/blob/release/v2/Boltzdesign1.ipynb

---

## 🚀 Quick Start
### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yehlincho/BoltzDesign1.git
   cd BoltzDesign1
   ```

2. **Run the automated setup**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

> ⚠️ **Note**: AlphaFold3 setup not included. Install separately following [official instructions](https://github.com/google-deepmind/alphafold3)

The setup script will automatically:
- ✅ Create conda environment with Python 3.10
- ✅ Install all required dependencies
- ✅ Set up Jupyter kernel for notebooks
- ✅ Download Boltz model weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta
- ❌ Need to install AF3 separately

---

## Run Code End-to-End
Run the complete pipeline from BoltzDesign to LigandMPNN/ProteinMPNN redesign and AlphaFold3 cross-validation.

Both **Boltz-2** (default) and **Boltz-1** are supported via `--boltz_model_version {boltz2,boltz1}`.
See `run_examples.ipynb` / `run_examples.py` for a runnable example of every target type on both models.

Small molecule (Boltz-2, the default):
```bash
python boltzdesign.py --name 7v11 --target_type small_molecule --target_seq OQO \
  --gpu_id 0 --design_samples 2 --suffix 1
```

Same target on Boltz-1:
```bash
python boltzdesign.py --name 7v11 --target_type small_molecule --target_seq OQO \
  --gpu_id 0 --design_samples 2 --suffix 1 --boltz_model_version boltz1
```

Protein target with MSA:
```bash
python boltzdesign.py --name 8znl --target_type protein --pdb_target_ids A \
  --target_seq FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK \
  --use_msa True --gpu_id 0 --design_samples 2 --suffix 1
```

Protein target with a template (**Boltz-2 only** — Boltz-1 does not support templates):
```bash
python boltzdesign.py --name 8znl --target_type protein --pdb_path 8znl --pdb_target_ids B \
  --use_template True --gpu_id 0 --design_samples 2 --suffix 1
```

DNA/RNA design:
```bash
python boltzdesign.py --name 5zmc --target_type dna \
  --target_seq GCCCTTCCGGGTCCCC,CGGGGACCCGGAAGGG --gpu_id 0 --design_samples 5 --suffix 1
```

Using your own PDB file:
```bash
python boltzdesign.py --name 7v11 --pdb_path your_pdb_path --target_type small_molecule \
  --target_seq OQO --gpu_id 0 --design_samples 2 --suffix own
```

### 🧪 AlphaFold3 cross-validation

Designs are validated with AlphaFold3 after LigandMPNN redesign. The validator is warm and
batched — the model is loaded once and reused across the batch, MSAs are cached — which is
roughly 4–5× faster end-to-end than launching a container per design, with the same model
and the same numbers.

AlphaFold3 is **not bundled**: its source is licensed CC BY-NC-SA 4.0 and its parameters
must be requested from DeepMind ([instructions](https://github.com/google-deepmind/alphafold3)).
It also needs its own conda environment, since AF3 runs on JAX/Python 3.11 while
BoltzDesign runs on PyTorch/Python 3.10. `setup.sh` creates it for you if an AlphaFold3
checkout is present.

- `--alphafold_dir`: your AlphaFold3 installation (default: `~/alphafold3`)
- `--af3_env_python`: python of the af3 env (default: `~/.conda/envs/af3/bin/python`, or `$AF3_ENV_PYTHON`)
- `--af3_num_diffusion_samples`: diffusion samples per design (default: 1)
- `--run_alphafold False`: skip validation entirely

See [docs/af3_validation.md](docs/af3_validation.md) for setup and outputs.

### 🔧 Additionally, you may need to optimize parameters for your binder/target:
- If binder does not form a highly compact structure, increase num_intra_contacts e.g. (default) 4 -> 8
- If target does not form interaction with binder, increase num_inter_contacts e.g. (default) 2 -> 4
- If generated binders have all alpha helices and you want to design beta sheets, lower the helix range, e.g. from the default `helix_loss_min -0.3 / helix_loss_max 0.0` to `-0.6 / -0.3` (more negative = less helix)
- If interaction features are not obtained through recycling=0, increase recycling_steps to 1 or more


## 🎥 Trajectory Visualization
We installed trajectory visualization based on LogMD
(https://github.com/log-md/logmd, implemented for Boltz diffusion trajectory https://colab.research.google.com/drive/1-9GXUPna4T0VFlDz9I64gzRQz259_G8f?usp=sharing#scrollTo=4eXNO1JJHYrB)

If you want to enable visualization of the trajectory, you need to set --save_trajectory True. However, be cautious that if you are just optimizing with distogram (--distogram_only True), it will take more time since it also runs the diffusion modules to get actual xyz coordinates.

---

## ⚙️ Design Configuration

Defaults live in per-target-type YAMLs under `boltzdesign/configs/`
(`default_sm_config.yaml`, `default_metal_config.yaml`, `default_na_config.yaml`,
`default_pep_config.yaml`, `default_ppi_config.yaml`). Any CLI flag you pass explicitly
overrides the config for that run.

```python
config = {
    # Optimization parameters
    'learning_rate': 0.1,       # Soft, temp, hard stages (protein target: 0.2)
    'learning_rate_pre': 0.2,   # Pre-iteration stage (only if pre_iteration > 0)
    # Iteration stages
    'pre_iteration': 0,         # Ligand-masked warm-up (0 = off; recommended)
    'soft_iteration': 75,       # Logits to Softmax optimization
    'temp_iteration': 45,       # Softmax temperature annealing
    'hard_iteration': 5,        # Final hard encoding optimization
    'semi_greedy_steps': 0,     # optional MCMC polish (off by default; see below)
    # Sequence initialization
    'sequence_init': 'gumbel',  # per-position softmax(scale * Gumbel) init
    'init_gumbel_scale': 1.0,   # 1.0 for ligand/metal/NA/peptide; 2.0 for protein targets
    # Secondary-structure bias (sampled uniformly per design)
    'helix_loss_min': -0.3,
    'helix_loss_max': 0.0,      # more negative = suppress helix -> push toward beta
    # Contacts
    'num_intra_contacts': 4,
    'num_inter_contacts': 2,
    # Algorithm settings
    'design_algorithm': '3stages',
}
```

> **Note on `semi_greedy_steps`**: optional, **off by default**. Each step runs 10 mutation
> trials scored by Boltz iPTM and keeps the best, so it reliably raises the Boltz confidence
> it optimizes (+0.03-0.05 iPTM on ~95% of designs in our runs) but costs roughly 2.5 min per
> design. Because it selects on the same score it reports, that gain is not independent
> evidence of a better binder - validate with AlphaFold3 either way. Enable with
> `--semi_greedy_steps 1`.

> **Note on `pre_iteration`**: the default is now **0**. The ligand-masked warm-up tends to
> seed elongated helical binders that the model cannot refold, which lowers validated
> success. Set `--pre_iteration 30` only to reproduce the original paper protocol.
---

## 🔄 Sequence Redesign

BoltzDesign1 supports sequence optimization using:

### ProteinMPNN
- **Use case**: Protein-protein interface design

### LigandMPNN  
- **Use case**: Protein-ligand and non-protein biomolecule interfaces

### Default setting
- Interface residues (< 4 Å) are **fixed** during design
- Non-interface residues are **redesigned**
- Custom interface definitions can be specified

---
## ✅ Structure Validation

### Primary Evaluation: AlphaFold3
Final structures are validated using **AlphaFold3** for:
- Structure quality assessment 
- Confidence scoring
- Cross-validation against design targets

### Alternative Options
- **Chai-1**: All-atom structure prediction
- **AlphaFold**: Protein monomer and multimer structure prediction

---

## 🎯 Successful Designs

After running the pipeline in `boltzdesign.py`, high-confidence designs can be found in:

`your_output_folder/ligandmpnn_cutoff_(interface threshold)/03_af_pdb_success`

The designs are saved along with `high_iptm_confidence_scores.csv`, which contains the iPTM and pLDDT scores for each design.

---

## 📋 Development Roadmap

### 🔬 Colab implementation
- [ ] **AlphaFold3 integration** for validation pipeline

### ⚡ Model Optimization  
- [ ] **Boltz1x Integration** 
- [ ] **Multi Chains Design** - Currently supporting single chain design
- [ ] **Multi-state optimization** - Alternating conformations
- [ ] **Specificity enhancement** - Target selectivity
### 🔧 Pipeline Features
- [ ] **RNA MSA Generation** - Multiple sequence alignments
  - Get Colab version of MSA extraction from ColabNuFold (https://colab.research.google.com/github/kiharalab/nufold/blob/master/ColabNuFold.ipynb#scrollTo=KDs4o5Bv35MI)
- [ ] **Input Support for DNA and RNA Modifications**
- [ ] **Advanced Filtering**:
  - [ ] Docking score integration
  - [ ] Metal coordination prediction
  - [ ] DNA/RNA specificity scoring
- [ ] **Enhanced Scoring**: Currently uses Rosetta scores (from [BindCraft])

## 📄 License & Citation

**License**: MIT License - See LICENSE file for details
**Citation**: If you use BoltzDesign1 in your research, please cite:
```
@article{cho2025boltzdesign1,
  title={Boltzdesign1: Inverting all-atom structure prediction model for generalized biomolecular binder design},
  author={Cho, Yehlin and Pacesa, Martin and Zhang, Zhidian and Correia, Bruno E and Ovchinnikov, Sergey},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
---

## 📧 Contact & Support

**Questions or Collaboration**: yehlin@mit.edu

**Issues**: Please report bugs and feature requests via GitHub Issues

---

## ⚠️ Important Disclaimer

> **EXPERIMENTAL SOFTWARE**: This pipeline is under active development and has **NOT been experimentally validated** in laboratory settings. We release this code to enable community contributions and collaborative development. Use at your own discretion and validate results independently.