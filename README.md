# BoltzDesign1 🧬

**BoltzDesign1** is a molecular design tool powered by the Boltz model for designing protein-protein interactions and biomolecular complexes.

> 📄 **Paper**: [BoltzDesign1: AI-Powered Molecular Design](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1)  
> 🚀 **Colab**: https://colab.research.google.com/github/yehlincho/BoltzDesign1/blob/main/Boltzdesign1.ipynb

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


Examle for small molecule:
python boltzdesign.py --target_name 7v11 --target_type small_molecule --target_mols OQO --gpu_id 0 --design_samples 2 --suffix 1

Example for DNA/RNA PDB design:
python boltzdesign.py --target_name 5zmc --target_type dna --pdb_target_ids C,D --gpu_id 0 --design_samples 5 --suffix 1

If you want to use your custom PDB file:
python boltzdesign.py --target_name 7v11 --pdb_path your_pdb_path --target_type small_molecule --target_mols OQO --gpu_id 0 --design_samples 2 --suffix own

⚠️ **Warning**: To run the AlphaFold3 cross-validation pipeline, you need to specify your AlphaFold3 directory, Docker name, database settings, and conda environment in the configuration. These can be set using the following arguments:
- `--alphafold_dir`: Path to your AlphaFold3 installation (default: ~/alphafold3)
- `--af3_docker_name`: Name of your AlphaFold3 Docker container
- `--af3_database_settings`: Path to AlphaFold3 database
- `--af3_hmmer_path`: Path to HMMER

If you want to disable af3 cross validation add flag --run_alphafold False

### 🔧 Additionally, you may need to optimize parameters for your binder/target:
- If binder does not form a highly compact structure, increase num_intra_contacts e.g. (default) 2 -> 4
- If target does not form interaction with binder, increase num_inter_contacts e.g. (default) 2 -> 4
- If generated binders have all alpha helices and you want to design beta sheets, change e.g. helix_loss_max 0.0, helix_loss_min = -0.3 to helix_loss_max -0.3, helix_loss_min = -0.6
- If interaction features are not obtained through recycling=0, increase recycling_steps to 1 or more


## 🎥 Trajectory Visualization
We installed trajectory visualization based on LogMD
(https://github.com/log-md/logmd, implemented for Boltz diffusion trajectory https://colab.research.google.com/drive/1-9GXUPna4T0VFlDz9I64gzRQz259_G8f?usp=sharing#scrollTo=4eXNO1JJHYrB)

If you want to enable visualization of the trajectory, you need to set --save_trajectory True. However, be cautious that if you are just optimizing with distogram (--distogram_only True), it will take more time since it also runs the diffusion modules to get actual xyz coordinates.

---

## ⚙️ Design Configuration

Configure your molecular design parameters:

```python
config = {
    # Optimization parameters
    'mutation_rate': 1,
    'learning_rate_pre': 0.2, ## Pre_iteration stage
    'learning_rate': 0.1, ## Soft, temp, hard stages
    # Iteration stages
    'pre_iteration': 30,      # Initial logits optimization
    'soft_iteration': 75,     # Logits to Softmax optimization
    'temp_iteration': 45,     # Softmax Temperature annealing
    'hard_iteration': 5,      # Final hard encoding optimization 
    'semi_greedy_steps': 0,   # MCMC based on iPTM score
    # Algorithm settings
    'design_algorithm': '3stages',
}
```
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