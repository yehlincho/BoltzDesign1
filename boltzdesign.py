import os
import sys
import argparse
import yaml
import json
import shutil
import pickle
import glob
import numpy as np
import random
import logging
import subprocess
import pandas as pd
from pathlib import Path

import sys
sys.path.append(f'{os.getcwd()}/boltzdesign')

from boltzdesign_utils import *
from ligandmpnn_utils import *
from alphafold_utils import *
from input_utils import *
from utils import *
import torch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_gpu_environment(gpu_id):
    """Setup GPU environment variables"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BoltzDesign: Protein Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design binder for DNA target
  python boltzdesign_generalized.py --target_name 5zmc --target_type dna --pdb_target_ids C,D --target_mols SAM --binder_id A
        """
    )
    
    # Required arguments
    parser.add_argument('--target_name', type=str, required=True,
                        help='Target name/PDB code (e.g., 5zmc)')
    # Target configuration
    parser.add_argument('--target_type', type=str, choices=['protein', 'rna', 'dna', 'small_molecule', 'metal'],
                        default='protein', help='Type of target molecule')
    parser.add_argument('--input_type', type=str, choices=['pdb', 'custom'], default='pdb',
                        help='Input type: pdb code or custom input')
    parser.add_argument('--pdb_target_ids', type=str, default='',
                        help='Target PDB IDs (comma-separated, e.g., "C,D")')
    parser.add_argument('--target_mols', type=str, default='',
                        help='Target molecules for small molecules (comma-separated, e.g., "SAM,FAD")')
    parser.add_argument('--custom_target_input', type=str, default='',
                        help='Custom target sequences/ligand(smiles)/dna/rna/metals (comma-separated, e.g., "ATAT,GCGC", "[O-]C(=O)C(N)CC[S+](C)CC3OC(n2cnc1c(ncnc12)N)C(O)C3O", "ZN")')
    parser.add_argument('--custom_target_ids', type=str, default='',
                        help='Custom target IDs (comma-separated, e.g., "A,B")')
    parser.add_argument('--binder_id', type=str, default='A',
                        help='Binder chain ID')
    parser.add_argument('--use_msa', type=str2bool, default=False,
                        help='Use MSA (if False, runs in single-sequence mode)')
    parser.add_argument('--msa_max_seqs', type=int, default=4096,
                        help='Maximum MSA sequences')
    parser.add_argument('--suffix', type=str, default='0',
                        help='Suffix for the output directory')
    
    # Modifications
    parser.add_argument('--modifications', type=str, default='',
                        help='Modifications (comma-separated)')
    parser.add_argument('--modifications_positions', type=str, default='',
                        help='Modification positions (comma-separated, matching order)')
    parser.add_argument('--modification_target', type=str, default='',
                        help='Target ID for modifications (e.g., "A")')
    
    # Constraints
    parser.add_argument('--constraint_target', type=str, default='',
                        help='Target ID for constraints (e.g., "A")')
    parser.add_argument('--contact_residues', type=str, default='',
                        help='Contact residues for constraints (comma-separated, e.g., "99,100,109")')

    
    # Design parameters
    parser.add_argument('--length_min', type=int, default=100,
                        help='Minimum binder length')
    parser.add_argument('--length_max', type=int, default=150,
                        help='Maximum binder length')
    parser.add_argument('--optimizer_type', type=str, choices=['SGD', 'AdamW'], default='SGD',
                        help='Optimizer type')
    
    # Iteration parameters
    parser.add_argument('--pre_iteration', type=int, default=30,
                        help='Pre-iteration steps')
    parser.add_argument('--soft_iteration', type=int, default=75,
                        help='Soft iteration steps')
    parser.add_argument('--temp_iteration', type=int, default=50,
                        help='Temperature iteration steps')
    parser.add_argument('--hard_iteration', type=int, default=5,
                        help='Hard iteration steps')
    parser.add_argument('--semi_greedy_steps', type=int, default=2,
                        help='Semi-greedy steps')
    
    # Advanced configuration
    parser.add_argument('--use_default_config', type=str2bool, default=True,
                        help='Use default configuration (recommended)')
    parser.add_argument('--mask_ligand', type=str2bool, default=False,
                        help='Mask target for warm-up stage')
    parser.add_argument('--optimize_contact_per_binder_pos', type=str2bool, default=False,
                        help='Optimize interface contact per binder position')
    parser.add_argument('--distogram_only', type=str2bool, default=True,
                        help='Only use distogram for optimization')
    parser.add_argument('--design_algorithm', type=str, choices=['3stages', '3stages_extra'], 
                        default='3stages', help='Design algorithm')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for optimization')
    parser.add_argument('--learning_rate_pre', type=float, default=0.1, 
                        help='Learning rate for pre iterations (warm-up stage)')
    parser.add_argument('--e_soft', type=float, default=0.8,
                        help='Softmax temperature for 3stages')
    parser.add_argument('--e_soft_1', type=float, default=0.8,
                        help='Initial softmax temperature for 3stages_extra')
    parser.add_argument('--e_soft_2', type=float, default=1.0,
                        help='Additional softmax temperature for 3stages_extra')
    
    # Interaction parameters
    parser.add_argument('--inter_chain_cutoff', type=int, default=20,
                        help='Inter-chain distance cutoff')
    parser.add_argument('--intra_chain_cutoff', type=int, default=14,
                        help='Intra-chain distance cutoff')
    parser.add_argument('--num_inter_contacts', type=int, default=2,
                        help='Number of inter-chain contacts')
    parser.add_argument('--num_intra_contacts', type=int, default=4,
                        help='Number of intra-chain contacts')
    
    # Helix loss parameters
    parser.add_argument('--helix_loss_max', type=float, default=0.0,
                        help='Maximum helix loss weight')
    parser.add_argument('--helix_loss_min', type=float, default=-0.3,
                        help='Minimum helix loss weight')
    
    # LigandMPNN parameters
    parser.add_argument('--num_designs', type=int, default=2,
                        help='Number of designs per PDB for LigandMPNN')
    parser.add_argument('--cutoff', type=int, default=4,
                        help='Cutoff distance for interface residues (Angstroms)')
    parser.add_argument('--i_ptm_cutoff', type=float, default=0.5,
                        help='iPTM cutoff for redesign')
    parser.add_argument('--complex_plddt_cutoff', type=float, default=0.7,
                        help='Complex pLDDT cutoff for high confidence designs')
    
    # System configuration
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--design_samples', type=int, default=1,
                        help='Number of design samples')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Working directory (default: current directory)')
    parser.add_argument('--high_iptm', type=str2bool, default=True,
                        help='Disable high iPTM designs')
    # Paths
    parser.add_argument('--boltz_checkpoint', type=str,
        default='~/.boltz/boltz1_conf.ckpt',
        help='Path to Boltz checkpoint')
    parser.add_argument('--ccd_path', type=str,
        default='~/.boltz/ccd.pkl',
        help='Path to CCD file')
    parser.add_argument('--alphafold_dir', type=str,
        default='~/alphafold3',
        help='AlphaFold directory')
    parser.add_argument('--af3_docker_name', type=str,
        default='alphafold3_yc',
        help='Docker name')
    # Control flags
    parser.add_argument('--run_boltz_design', type=str2bool, default=True,
                        help='Run Boltz design step')
    parser.add_argument('--run_ligandmpnn', type=str2bool, default=True,
                        help='Run LigandMPNN redesign step')
    parser.add_argument('--run_alphafold', type=str2bool, default=True,
                        help='Run AlphaFold validation step')
    parser.add_argument('--run_rosetta', type=str2bool, default=True,
                        help='Run Rosetta energy calculation (protein targets only)')
    parser.add_argument('--redo_boltz_predict', type=str2bool, default=False,
                        help='Redo Boltz prediction')


    ## Visualization
    parser.add_argument('--show_animation', type=str2bool, default=True,
                        help='Show animation')
    parser.add_argument('--save_trajectory', type=str2bool, default=False,
                        help='Save trajectory')
    return parser.parse_args()


class YamlConfig:
    """Configuration class for managing directories"""
    def __init__(self, main_dir: str = None):
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / 'inputs'
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / 'PDB'
        self.MSA_DIR = self.MAIN_DIR / 'MSA'
        self.YAML_DIR = self.MAIN_DIR / 'yaml'
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


def load_boltz_model(checkpoint_path, device):
    """Load Boltz model"""
    predict_args = {
        "recycling_steps": 1,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    boltz_model = get_boltz_model(checkpoint_path, predict_args, device)
    boltz_model.train()
    return boltz_model, predict_args

def process_modifications(modifications, modifications_positions):
    """Process modifications data"""
    if modifications and modifications_positions:
        mod_list = [mod.strip() for mod in modifications.split(',')]
        pos_list = [int(pos.strip()) for pos in modifications_positions.split(',')]
        
        if len(mod_list) != len(pos_list):
            raise ValueError("Number of modifications and positions must match.")
        
        modifications_data = []
        for mod, pos in zip(mod_list, pos_list):
            modifications_data.append({
                'position': pos,
                'ccd': mod
            })
    else:
        modifications_data = None
    return modifications_data

def setup_constraints(contact_residues, binder_id, target_id):
    """Setup binding constraints"""
    constraints = None
    if contact_residues:
        residues = [int(x.strip()) for x in contact_residues.split(",")]
        constraints = {
            'pocket': {
                'binder': binder_id,
                'contacts': [[target_id, res] for res in residues]
            }
        }
    return constraints

def load_design_config(target_type, work_dir):
    """Load design configuration based on target type"""
    if target_type in ('small_molecule', 'metal'):
        config_path = f"{work_dir}/boltzdesign/configs/default_sm_config.yaml"
    elif target_type in ('dna', 'rna'):
        config_path = f"{work_dir}/boltzdesign/configs/default_na_config.yaml"
    elif target_type == 'protein':
        config_path = f"{work_dir}/boltzdesign/configs/default_ppi_config.yaml"
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_explicit_args():
    # Get all command-line arguments (excluding the script name)
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Handle --arg=value and --arg value
            if '=' in arg:
                explicit_args.add(arg.split('=')[0].lstrip('-').replace('-', '_'))
            else:
                explicit_args.add(arg.lstrip('-').replace('-', '_'))
    return explicit_args

def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    # Always update these basic parameters regardless of use_default_config
    basic_params = {
        'binder_chain': args.binder_id,
        'non_protein_target': args.target_type != 'protein',
        'msa_max_seqs': args.msa_max_seqs,
        'length_min': args.length_min,
        'length_max': args.length_max,
        'optimizer_type': args.optimizer_type,
        'pocket_conditioning': bool(args.contact_residues),
        'pre_iteration': args.pre_iteration,
        'soft_iteration': args.soft_iteration,
        'temp_iteration': args.temp_iteration,
        'hard_iteration': args.hard_iteration,
        'semi_greedy_steps': args.semi_greedy_steps
    }
    
    # Update basic parameters
    explicit_args = get_explicit_args()
    config.update(basic_params)
    
    # For advanced parameters, only update those that are explicitly set by the user
    # (i.e., different from their default values in argparse)
    parser = argparse.ArgumentParser()
    _, defaults = parser.parse_known_args([])  # Get default values
    
    advanced_params = {
        'mask_ligand': args.mask_ligand,
        'optimize_contact_per_binder_pos': args.optimize_contact_per_binder_pos,
        'distogram_only': args.distogram_only,
        'design_algorithm': args.design_algorithm,
        'learning_rate': args.learning_rate,
        'learning_rate_pre': args.learning_rate_pre,
        'e_soft': args.e_soft,
        'e_soft_1': args.e_soft_1,
        'e_soft_2': args.e_soft_2,
        'inter_chain_cutoff': args.inter_chain_cutoff,
        'intra_chain_cutoff': args.intra_chain_cutoff,
        'num_inter_contacts': args.num_inter_contacts,
        'num_intra_contacts': args.num_intra_contacts,
        'helix_loss_max': args.helix_loss_max,
        'helix_loss_min': args.helix_loss_min
    }
    
    # Only update parameters that were explicitly set (different from defaults)
    # if not args.use_default_config:
    # for param_name, param_value in advanced_params.items():
    #     arg_name = f"--{param_name}"
    #     if hasattr(args, param_name) and getattr(args, param_name) != getattr(defaults, param_name, None):
    #         print(f"Updating {param_name} to {param_value}")
    #         config[param_name] = param_value
    for param_name, param_value in advanced_params.items():
        if param_name in explicit_args:
            print(f"Updating {param_name} to {param_value}")
            config[param_name] = param_value
    return config
    

def run_boltz_design_step(args, config, boltz_model, yaml_dir, main_dir, version_name):
    """Run the Boltz design step"""
    print("Starting Boltz design step...")
    
    loss_scales = {
        'con_loss': 1.0,
        'i_con_loss': 1.0,
        'plddt_loss': 0.1,
        'pae_loss': 0.4,
        'i_pae_loss': 0.1,
        'rg_loss': 0.0,
    }
    
    boltz_path = shutil.which("boltz")
    if boltz_path is None:
        raise FileNotFoundError("The 'boltz' command was not found in the system PATH.")
    
    run_boltz_design(
        boltz_path=boltz_path,
        main_dir=main_dir,
        yaml_dir=os.path.dirname(yaml_dir),
        boltz_model=boltz_model,
        ccd_path=args.ccd_path,
        design_samples=args.design_samples,
        version_name=version_name,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        redo_boltz_predict=args.redo_boltz_predict,
    )
    
    print("Boltz design step completed!")

def run_ligandmpnn_step(args, main_dir, version_name, yaml_dir, work_dir):
    """Run the LigandMPNN redesign step"""
    print("Starting LigandMPNN redesign step...")
    
    # Setup LigandMPNN config
    yaml_path = f"{work_dir}/LigandMPNN/run_ligandmpnn_logits_config.yaml"
    with open(yaml_path, "r") as f:
        mpnn_config = yaml.safe_load(f)
    
    for key, value in mpnn_config.items():
        if isinstance(value, str) and "${CWD}" in value:
            mpnn_config[key] = value.replace("${CWD}", work_dir)
    
    if not Path(mpnn_config["checkpoint_soluble_mpnn"]).exists():
        raise FileNotFoundError("LigandMPNN checkpoint file not found!")
    
    with open(yaml_path, "w") as f:
        yaml.dump(mpnn_config, f, default_flow_style=False)
    
    # Setup directories
    boltzdesign_dir = f"{main_dir}/{version_name}/results_final"
    pdb_save_dir = f"{main_dir}/{version_name}/pdb"
    ligandmpnn_dir = f"{main_dir}/{version_name}/ligandmpnn_cutoff"
    
    lmpnn_redesigned_dir = os.path.join(ligandmpnn_dir, 'lmpnn_redesigned')
    lmpnn_redesigned_fa_dir = os.path.join(ligandmpnn_dir, 'lmpnn_redesigned_fa')
    lmpnn_redesigned_yaml_dir = os.path.join(ligandmpnn_dir, 'lmpnn_redesigned_yaml')
    
    os.makedirs(ligandmpnn_dir, exist_ok=True)
    # Convert CIF to PDB and run LigandMPNN
    convert_cif_files_to_pdb(boltzdesign_dir, pdb_save_dir, high_iptm=args.high_iptm, i_ptm_cutoff=args.i_ptm_cutoff)

    if not any(f.endswith('.pdb') for f in os.listdir(pdb_save_dir)):
        print("No successful designs from BoltzDesign")
        sys.exit(1)
    
    run_ligandmpnn_redesign(
        ligandmpnn_dir, pdb_save_dir, shutil.which("boltz"),
        os.path.dirname(yaml_dir), yaml_path, top_k=args.num_designs, cutoff=args.cutoff,
        non_protein_target=args.target_type != 'protein', binder_chain=args.binder_id,
        target_chains="all", out_dir=lmpnn_redesigned_fa_dir,
        lmpnn_yaml_dir=lmpnn_redesigned_yaml_dir, results_final_dir=lmpnn_redesigned_dir
    )
    
    # Filter high confidence designs
    filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir)
    
    print("LigandMPNN redesign step completed!")
    return ligandmpnn_dir

def filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir):
    """Filter and save high confidence designs"""
    print("Filtering high confidence designs...")
    
    yaml_dir_success_designs_dir = os.path.join(ligandmpnn_dir, 'lmpnn_redesigned_high_iptm')
    yaml_dir_success_boltz_yaml = os.path.join(yaml_dir_success_designs_dir, 'yaml')
    yaml_dir_success_boltz_cif = os.path.join(yaml_dir_success_designs_dir, 'cif')
    
    os.makedirs(yaml_dir_success_boltz_yaml, exist_ok=True)
    os.makedirs(yaml_dir_success_boltz_cif, exist_ok=True)
    
    successful_designs = 0
    
    # Process designs
    for root in os.listdir(lmpnn_redesigned_dir):
        root_path = os.path.join(lmpnn_redesigned_dir, root, 'predictions')
        if not os.path.isdir(root_path):
            continue
        
        for subdir in os.listdir(root_path):
            json_path = os.path.join(root_path, subdir, f'confidence_{subdir}_model_0.json')
            yaml_path = os.path.join(lmpnn_redesigned_yaml_dir, f'{subdir}.yaml')
            cif_path = os.path.join(lmpnn_redesigned_dir, f'boltz_results_{subdir}', 'predictions', subdir, f'{subdir}_model_0.cif')
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                design_name = json_path.split('/')[-2]
                length = int(subdir[subdir.find('length') + 6:subdir.find('_model')])
                iptm = data.get('iptm', 0)
                complex_plddt = data.get('complex_plddt', 0)
                
                print(f"{design_name} length: {length} complex_plddt: {complex_plddt:.2f} iptm: {iptm:.2f}")
                
                if iptm > args.i_ptm_cutoff and complex_plddt > args.complex_plddt_cutoff:
                    shutil.copy(yaml_path, os.path.join(yaml_dir_success_boltz_yaml, f'{subdir}.yaml'))
                    shutil.copy(cif_path, os.path.join(yaml_dir_success_boltz_cif, f'{subdir}.cif'))
                    print(f"✅ {design_name} copied")
                    successful_designs += 1
            
            except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Skipping {subdir}: {e}")
                continue
    
    if successful_designs == 0:
        print("Error: No LigandMPNN/ProteinMPNN redesigned designs passed the confidence thresholds")
        sys.exit(1)


def run_alphafold_step(args, ligandmpnn_dir, work_dir):
    """Run AlphaFold validation step"""
    print("Starting AlphaFold validation step...")

    alphafold_dir = os.path.expanduser(args.alphafold_dir)
    print("alphafold_dir", alphafold_dir)
    
    # Create AlphaFold directories
    af_input_dir = f'{ligandmpnn_dir}/design_json_af3'
    af_output_dir = f'{ligandmpnn_dir}/design_final_af3'
    af_input_apo_dir = f'{ligandmpnn_dir}/design_json_af3_apo'
    af_output_apo_dir = f'{ligandmpnn_dir}/design_final_af3_apo'
    
    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process YAML files
    yaml_dir_success_boltz_yaml = os.path.join(ligandmpnn_dir, 'lmpnn_redesigned_high_iptm', 'yaml')
    
    process_yaml_files(
        yaml_dir_success_boltz_yaml,
        af_input_dir,
        af_input_apo_dir,
        target_type=args.target_type,
        binder_chain=args.binder_id
    )
    
    # Run AlphaFold on holo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_dir,
        af_output_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    # Run AlphaFold on apo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_apo_dir,
        af_output_apo_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    print("AlphaFold validation step completed!")
    return af_output_dir, af_output_apo_dir

def run_rosetta_step(args, ligandmpnn_dir, af_output_dir, af_output_apo_dir):
    """Run Rosetta energy calculation (protein targets only)"""
    if args.target_type != 'protein':
        print("Skipping Rosetta step (not a protein target)")
        return
    
    print("Starting Rosetta energy calculation...")
    
    af_pdb_dir = f"{ligandmpnn_dir}/af_pdb"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/af_pdb_apo"
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"
    
    convert_cif_files_to_pdb(af_output_dir, af_pdb_dir, af_dir=True, high_iptm=args.high_iptm)
    if not any(f.endswith('.pdb') for f in os.listdir(af_pdb_dir)):
        print("No successful designs from AlphaFold")
        sys.exit(1)

    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)

    
    measure_rosetta_energy(
        af_pdb_dir, af_pdb_dir_apo, af_pdb_rosetta_success_dir,
        binder_holo_chain=args.binder_id, binder_apo_chain='A'
    )
    
    print("Rosetta energy calculation completed!")
    
def main():
    """Main function for running the BoltzDesign pipeline"""
    args = setup_environment()
    boltz_model, config_obj, target_id_map = initialize_pipeline(args)
    
    # Process constraints and modifications
    constraints, modifications = process_design_constraints(args, target_id_map)
    
    # Generate YAML configuration
    yaml_dict, yaml_dir = generate_yaml_config(args, config_obj, constraints, modifications)
    print("Generated YAML configuration:", yaml_dict)
    
    # Setup pipeline configuration
    config = setup_pipeline_config(args)
    output_dir = setup_output_directories(args)
    
    # Run pipeline steps
    print("config:")
    items = list(config.items())
    max_key_len = max(len(key) for key, _ in items)
    max_val_len = max(len(str(val)) for _, val in items)
    
    # Print header
    print("  " + "=" * (max_key_len + max_val_len + 5))
    
    # Print items in two columns
    for i in range(0, len(items), 2):
        key1, value1 = items[i]
        if i+1 < len(items):
            key2, value2 = items[i+1]
            print(f"  {key1:<{max_key_len}}: {str(value1):<{max_val_len}}    "
                  f"{key2:<{max_key_len}}: {value2}")
        else:
            print(f"  {key1:<{max_key_len}}: {value1}")
    
    print("  " + "=" * (max_key_len + max_val_len + 5))
    results = run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir)
    
    print("Pipeline completed successfully!")

def setup_environment():
    """Setup environment and parse arguments"""
    args = parse_arguments()
    work_dir = args.work_dir or os.getcwd()
    os.chdir(work_dir)
    setup_gpu_environment(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args

def initialize_pipeline(args):
    """Initialize models and configurations"""
    work_dir = args.work_dir or os.getcwd()
    boltz_model, predict_args = load_boltz_model(args.boltz_checkpoint, 
                                               torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    config_obj = YamlConfig(main_dir=f'{work_dir}/inputs/test_data/{args.target_name}_binder')
    config_obj.setup_directories()
    
    # Map target chain IDs
    target_ids_list = get_target_ids(args)
    target_id_map = map_target_chain_ids(target_ids_list, args.binder_id)
    print(f"Mapped target IDs: {list(target_id_map.values())}")
    
    return boltz_model, config_obj, target_id_map

def process_design_constraints(args, target_id_map):
    """Process design constraints and modifications"""
    if not (args.contact_residues or args.modifications):
        return None, None
        
    constraint_target = target_id_map.get(args.constraint_target, '') if args.contact_residues else ''
    modification_target = target_id_map.get(args.modification_target, '') if args.contact_residues else ''
    
    modifications = {
        'data': process_modifications(args.modifications, args.modifications_positions),
        'target': modification_target
    }
    constraints = setup_constraints(args.contact_residues, args.binder_id, constraint_target)
    
    return constraints, modifications

def generate_yaml_config(args, config_obj, constraints, modifications):
    """Generate YAML configuration based on input type"""
    if args.input_type == "pdb":
        pdb_target_ids = [str(x.strip()) for x in args.pdb_target_ids.split(",")] if args.pdb_target_ids else None
        target_mols = [str(x.strip()) for x in args.target_mols.split(",")] if args.target_mols else None
        
        return generate_yaml_from_pdb(
            args.target_name,
            args.target_type,
            config_obj,
            binder_id=args.binder_id,
            target_ids=pdb_target_ids,
            target_mols=target_mols,
            constraints=constraints,
            modifications=modifications['data'] if modifications else None,
            modification_target=modifications['target'] if modifications else None,
            use_msa=args.use_msa
        )
    
    target_inputs = [str(x.strip()) for x in args.custom_target_input.split(",")] if args.custom_target_input else []
    target = target_inputs or [args.target_name]
    return generate_custom_yaml(
        args.target_name,
        args.target_type,
        target,
        config_obj,
        binder_id=args.binder_id,
        constraints=constraints,
        modifications=modifications['data'] if modifications else None,
        modification_target=modifications['target'] if modifications else None,
        use_msa=args.use_msa
    )

def setup_pipeline_config(args):
    """Setup pipeline configuration"""
    work_dir = args.work_dir or os.getcwd()
    config = load_design_config(args.target_type, work_dir)
    return update_config_with_args(config, args)

def setup_output_directories(args):
    """Setup output directories"""
    work_dir = args.work_dir or os.getcwd()
    main_dir = f'{work_dir}/outputs'
    os.makedirs(main_dir, exist_ok=True)
    return {
        'main_dir': main_dir,
        'version': f'{args.target_type}_{args.target_name}_{args.suffix}'
    }

def run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir):
    """Run the pipeline steps based on arguments"""
    results = {'ligandmpnn_dir': None, 'af_output_dir': None, 'af_output_apo_dir': None}
    
    if args.run_boltz_design:
        print("Running BoltzDesign step...")
        run_boltz_design_step(args, config, boltz_model, yaml_dir, 
                            output_dir['main_dir'], output_dir['version'])
    
    if args.run_ligandmpnn:
        print("Running LigandMPNN step...")
        results['ligandmpnn_dir'] = run_ligandmpnn_step(
            args, output_dir['main_dir'], output_dir['version'], 
            yaml_dir, args.work_dir or os.getcwd()
        )
    
    if args.run_alphafold:
        print("Running AlphaFold step...")
        results['af_output_dir'], results['af_output_apo_dir'] = run_alphafold_step(
            args, results['ligandmpnn_dir'], args.work_dir or os.getcwd()
        )
    
    if args.run_rosetta:
        print("Running Rosetta step...")
        run_rosetta_step(args, results['ligandmpnn_dir'], 
                        results['af_output_dir'], results['af_output_apo_dir'])
    
    return results

def get_target_ids(args):
    """Get target IDs based on input type"""
    if args.input_type == "pdb":
        return [str(x.strip()) for x in args.pdb_target_ids.split(",")]
    elif args.input_type == "custom":
        return [str(x.strip()) for x in args.custom_target_ids.split(",")]
    raise ValueError("Invalid input type")

def map_target_chain_ids(target_ids_list, binder_id):
    """Map original target IDs to new chain IDs"""
    target_id_map = {}
    for i, orig_id in enumerate(target_ids_list):
        new_id = chr(ord('A') + i + (1 if ord('A') + i >= ord(binder_id) else 0))
        target_id_map[orig_id] = new_id
    return target_id_map

if __name__ == "__main__":
    main()
