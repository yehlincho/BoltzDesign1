# BoltzDesign1 🧬

**BoltzDesign1** is a molecular design tool powered by the Boltz model for designing protein-protein interactions and biomolecular complexes.

> 📄 **Paper**: [BoltzDesign1: AI-Powered Molecular Design](https://www.biorxiv.org/content/10.1101/2025.04.06.647261v1)  
> 🚀 **Colab**: *Coming Soon*

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

The setup script will automatically:
- ✅ Create conda environment with Python 3.10
- ✅ Install all required dependencies
- ✅ Set up Jupyter kernel for notebooks
- ✅ Download Boltz model weights
- ✅ Configure LigandMPNN and ProteinMPNN
- ✅ Optionally install PyRosetta

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

### Default Behavior
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

> ⚠️ **Note**: AlphaFold3 setup not included. Install separately following [official instructions](https://github.com/google-deepmind/alphafold3)

### Alternative Options
- **Chai-1**: Fast structure prediction
- **ColabFold**: Accessible online validation

---

## 📋 Development Roadmap

### 🔬 Colab implementation
- [ ] **AlphaFold3 integration** for validation pipeline

### ⚡ Model Optimization  
- [ ] **Boltz1x integration** - Next-generation model
- [ ] **Multi Chains Design** - Currently supporting single chain design
- [ ] **Multi-state optimization** - Alternating conformations
- [ ] **Specificity enhancement** - Target selectivity

### 🔧 Pipeline Features
- [ ] **RNA MSA generation** - Multiple sequence alignments
- [ ] **Advanced filtering**:
  - [ ] Docking score integration
  - [ ] Metal coordination prediction
  - [ ] DNA/RNA specificity scoring
- [ ] **Enhanced scoring**: Currently uses Rosetta scores (from [BindCraft])

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