# BoltzDesign1

BoltzDesign1 is a molecular design tool powered by the Boltz model. This README provides a comprehensive guide for setting up and using the tool, with examples available in the companion notebook.

## Setup

### Environment Configuration
Before running BoltzDesign1, ensure you have configured your environment properly with all required dependencies.
Boltz code and weights are provided under MIT license.

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)



### Dependencies
Required packages:
- PyTorch (`torch`) - Deep learning framework
- RDKit (`rdkit`) - Molecular operations and visualization
- boltz-1
- `boltzdesign_utils` - Custom utilities for Boltz model operations

### Model Configuration
The Boltz model uses default prediction arguments during initialization. These can be customized as needed.

default is 

predict_args = {
"recycling_steps": 0,
"sampling_steps": 200,
"diffusion_samples": 1,
"write_confidence_summary": True,
"write_full_pae": False,
"write_full_pde": False,
}

## Design Configuration
The molecular design process is controlled by these configuration parameters:
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

## Sequence Redesign
Sequence can be redesign with ProteinMPNN for PPI and LigandMPNN for non-protein biomolecule binders. 
Default setting is fix interface residue CA distance < 5 A and redesign rest part of it.


## Final Evaluation
We utilized AF3 for final validation. But other model can be used (Chai,...)