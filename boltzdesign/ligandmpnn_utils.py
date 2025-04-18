import os


import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule

import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

from boltz.data.parse.fasta import parse_fasta
from boltz.main import BoltzProcessedInput, BoltzDiffusionParams
from boltz.data.types import MSA, Manifest
from boltz.data.module.inference import BoltzInferenceDataModule
from rdkit import Chem
from boltz.data.feature import featurizer
from boltz.data.tokenize.boltz import BoltzTokenizer, TokenData
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure
from boltz.data.parse.yaml import parse_yaml



from matplotlib.animation import FuncAnimation
import numpy as np

from matplotlib.animation import FuncAnimation
import numpy as np
from boltz.data.parse.schema import parse_boltz_schema
import yaml
from pathlib import Path
# Add project root and LigandMPNN to Python path
import os
import sys

# Get the project root directory (parent of boltzdesign)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root and LigandMPNN to path if not already there
if project_root not in sys.path:
    sys.path.append(project_root)
if os.path.join(project_root, 'LigandMPNN') not in sys.path:
    sys.path.append(os.path.join(project_root, 'LigandMPNN'))

from prody import parsePDB
import numpy as np

import yaml
from pathlib import Path
from run import main
from argparse import Namespace
from types import SimpleNamespace


import sys
import logging
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO

def int_to_chain(i,base=62):
    """
    int_to_chain(int,int) -> str

    Converts a positive integer to a chain ID. Chain IDs include uppercase
    characters, numbers, and optionally lowercase letters.

    i = a positive integer to convert
    base = the alphabet size to include. Typically 36 or 62.
    """
    if i < 0:
        raise ValueError("positive integers only")
    if base < 0 or 62 < base:
        raise ValueError("Invalid base")

    quot = int(i)//base
    rem = i%base
    if rem < 26:
        letter = chr( ord("A") + rem)
    elif rem < 36:
        letter = str( rem-26)
    else:
        letter = chr( ord("a") + rem - 36)
    if quot == 0:
        return letter
    else:
        return int_to_chain(quot-1,base) + letter

class OutOfChainsError(Exception): pass
def rename_chains(structure):
    """Renames chains to be one-letter chains
    
    Existing one-letter chains will be kept. Multi-letter chains will be truncated
    or renamed to the next available letter of the alphabet.
    
    If more than 62 chains are present in the structure, raises an OutOfChainsError
    
    Returns a map between new and old chain IDs, as well as modifying the input structure
    """
    next_chain = 0 #
    # single-letters stay the same
    chainmap = {c.id:c.id for c in structure.get_chains() if len(c.id) == 1}
    for o in structure.get_chains():
        if len(o.id) != 1:
            if o.id[0] not in chainmap:
                chainmap[o.id[0]] = o.id
                o.id = o.id[0]
            else:
                c = int_to_chain(next_chain)
                while c in chainmap:
                    next_chain += 1
                    c = int_to_chain(next_chain)
                    if next_chain >= 62:
                        raise OutOfChainsError()
                chainmap[c] = o.id
                o.id = c
    return chainmap

def convert_cif_to_pdb(ciffile, pdbfile):
    """
    Convert a CIF file to PDB format, handling chain renaming.
    
    Args:
        ciffile (str): Path to input CIF file
        pdbfile (str): Path to output PDB file
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARN)

    #Not sure why biopython needs this to read a cif file
    strucid = ciffile[:4] if len(ciffile)>4 else "1xxx"

    # Read file
    parser = MMCIFParser()
    structure = parser.get_structure(strucid, ciffile)

    # rename long chains
    try:
        chainmap = rename_chains(structure)
    except OutOfChainsError:
        logging.error("Too many chains to represent in PDB format")
        return False

    #Write PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile)
    return True

import os

def convert_cif_files_to_pdb(results_dir, save_dir):
    """
    Convert all .cif files in results_dir to .pdb format and save in save_dir
    
    Args:
        results_dir (str): Directory containing .cif files
        save_dir (str): Directory to save converted .pdb files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.cif'):
                cif_path = os.path.join(root, file)
                pdb_path = os.path.join(save_dir, file.replace('.cif', '.pdb'))
                print(f"Converting {cif_path}")
                convert_cif_to_pdb(cif_path, pdb_path)

# Set up paths



def get_protein_ligand_interface(pdb_id, cutoff=6, non_protein_ligand=True, binder_chain='A', target_chain='B'):
    """
    Get interface residues between a protein and ligand/target from a PDB structure.
    
    Args:
        pdb_id (str): Path to PDB file
        cutoff (float): Distance cutoff in Angstroms for interface residues
        non_protein_ligand (bool): Whether to select non-protein ligand (True) or protein target chain (False)
        binder_chain (str): Chain ID of the protein binder chain
        target_chain (str): Chain ID of the target protein chain (if non_protein_ligand=False)
    
    Returns:
        list: Residue indices that are at the interface
    """
    # Load PDB structure
    pdb = parsePDB(pdb_id)
    
    # Get binder protein coordinates
    protein = pdb.select(f'protein and chain {binder_chain} and name CA')
    protein_coords = protein.getCoords()
    
    # Get ligand/target coordinates
    if non_protein_ligand:
        ligand = pdb.select('not protein and not water')
    else:
        ligand = pdb.select(f'protein and chain {target_chain} and name CA')
    ligand_coords = ligand.getCoords()
    
    # Calculate pairwise distances and find interface residues
    distances = np.sqrt(np.sum((protein_coords[:,None,:] - ligand_coords[None,:,:])**2, axis=-1))
    interacting = distances < cutoff
    interface_residues = list(np.nonzero(np.sum(interacting, axis=-1))[0])
    
    return interface_residues


def run_ligandmpnn_redesign(base_dir, pdb_dir, yaml_dir, ligandmpnn_config, top_k=5, cutoff=6, non_protein_ligand=True, binder_chain='A', target_chain='B'):
    out_dir = os.path.join(base_dir, 'boltz_hallucination_success_lmpnn_fa')
    lmpnn_yaml_dir = os.path.join(base_dir, 'boltz_hallucination_success_lmpnn_yaml')
    results_final_dir = os.path.join(base_dir, 'boltz_predictions_success_lmpnn')

    # Create required directories
    for directory in [out_dir, lmpnn_yaml_dir, results_final_dir]:
        os.makedirs(directory, exist_ok=True)

    # Initialize score tracking lists
    original_score = []
    ligandpmpnn_redesign_score = []

    for pdb_path in os.listdir(pdb_dir):
        pdb_name = pdb_path.split('.pdb')[0]
        existing_yamls = list(Path(lmpnn_yaml_dir).glob(f'{pdb_name}_*.yaml'))
        if existing_yamls:
            print(f"Skipping {pdb_name} as yaml files already exist")
            continue
        else:
            pdb_path = os.path.join(pdb_dir, pdb_path)
            if pdb_path.endswith('.pdb'):
                interface_residues= get_protein_ligand_interface(pdb_path, cutoff=cutoff, non_protein_ligand=non_protein_ligand, binder_chain=binder_chain, target_chain=target_chain)
                print("len interface_residues", len(interface_residues))
                with open(ligandmpnn_config, 'r') as f:
                    config_dict = yaml.safe_load(f)

                if non_protein_ligand:
                    model_type = "ligand_mpnn"
                else:
                    model_type = "protein_mpnn"

                config = SimpleNamespace(**config_dict)
                config.model_type = model_type
                config.seed = 111
                config.pdb_path = pdb_path
                config.out_folder = out_dir
                config.fixed_residues = " ".join([f'{binder_chain}{item+1}' for item in interface_residues])
                config.batch_size = 16
                config.save_stats = 0
                config.chains_to_design = binder_chain
                output = main(config)
                fasta_path = os.path.join(out_dir, 'seqs', f'{pdb_name}.fa')
                print(fasta_path)
                # Read the existing fa file
                with open(fasta_path, 'r') as f:
                    lines = f.readlines()
                    sequences = []
                    sequence_found = False  # Flag to check if sequence is found
                    for line in lines[2:]:
                        if line.startswith('>'):
                            overall_confidence = float(line.split(',')[4].split('=')[1])
                            ligand_confidence = line.split(',')[5].split('=')[1]
                            sequences.append((overall_confidence, ligand_confidence, ""))  # Store confidence and ligand
                            sequence_found = True  # Set flag to true if sequence is found
                        elif sequence_found:  # Check for the sequence line after finding confidence
                            sequences[-1] = (sequences[-1][0], sequences[-1][1], line.strip())  # Add the corresponding sequence
                            sequence_found = False  # Reset flag after capturing the sequence

                top_sequences = sorted(sequences, key=lambda x: x[0], reverse=True)[:top_k]
                for idx, (overall_confidence, ligand_confidence, sequence) in enumerate(top_sequences):
                    matching_yamls = list(Path(yaml_dir).glob(f'{pdb_name.split("_results")[0]}*.yaml'))
                    if matching_yamls:
                        yaml_path = str(matching_yamls[0])  # Take the first matching yaml file
                        with open(yaml_path, 'r') as f:
                            yaml_data = yaml.safe_load(f)
                    
                    with open(yaml_path, 'r') as f:
                        yaml_data = yaml.safe_load(f)


                    if not non_protein_ligand:
                        if target_chain =='A' and binder_chain == 'B':
                            msa_dir = yaml_data['sequences'][0]['protein']['msa']
                            yaml_data['sequences'][0]['protein']['msa'] = msa_dir.replace('.npz', '.a3m')
                            print("sequence", sequence)
                            yaml_data['sequences'][1]['protein']['sequence'] = sequence.split(':')[1]
                            yaml_data.pop('constraints', None)
                        elif target_chain =='B' and binder_chain == 'A':
                            msa_dir = yaml_data['sequences'][1]['protein']['msa']
                            yaml_data['sequences'][1]['protein']['msa'] = msa_dir.replace('.npz', '.a3m')
                            print("sequence", sequence)
                            yaml_data['sequences'][0]['protein']['sequence'] = sequence.split(':')[0]
                            yaml_data.pop('constraints', None)

                    else:
                        if binder_chain == 'A':
                            yaml_data['sequences'][0]['protein']['sequence'] = sequence
                        elif binder_chain == 'B':
                            yaml_data['sequences'][1]['protein']['sequence'] = sequence
    

                    for key in yaml_data:
                        if isinstance(yaml_data[key], str) and yaml_data[key].endswith('.npz'):
                            yaml_data[key] = yaml_data[key][:-4] + '.a3m'
                        elif isinstance(yaml_data[key], dict):
                            for subkey in yaml_data[key]:
                                if isinstance(yaml_data[key][subkey], str) and yaml_data[key][subkey].endswith('.npz'):
                                    yaml_data[key][subkey] = yaml_data[key][subkey][:-4] + '.a3m'


                    print("yaml_data")
                    print(yaml_data)
                    
                    final_yaml_path = os.path.join(lmpnn_yaml_dir, f'{pdb_name}_{idx+1}.yaml')
                    print("final_yaml_path") 
                    print(final_yaml_path)
                    with open(final_yaml_path, 'w') as f:
                        yaml.dump(yaml_data, f)

                    import subprocess
                    subprocess.run(['boltz', 'predict', str(final_yaml_path), '--out_dir', str(results_final_dir), '--write_full_pae'])
                    print(f"Completed processing {pdb_name} for sequence {idx+1}")



