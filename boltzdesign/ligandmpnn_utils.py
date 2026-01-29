import json
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional
import numpy as np
import pandas as pd
import torch
import yaml
from Bio.PDB import PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser
from prody import parsePDB

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ligandmpnn_path = os.path.join(project_root, 'LigandMPNN')

if project_root not in sys.path:
    sys.path.append(project_root)
if ligandmpnn_path not in sys.path:
    sys.path.append(ligandmpnn_path)

from run import main

chain_to_number = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
}

def int_to_chain(i, base=62):
    """
    Converts a positive integer to a PDB-valid chain ID using uppercase letters,
    numbers, and optionally lowercase letters (base up to 62).
    """
    if i < 0:
        raise ValueError("positive integers only")
    if base < 0 or 62 < base:
        raise ValueError("Invalid base")

    quot = int(i) // base
    rem = i % base
    if rem < 26:
        letter = chr(ord("A") + rem)
    elif rem < 36:
        letter = str(rem - 26)
    else:
        letter = chr(ord("a") + rem - 36)

    if quot == 0:
        return letter
    else:
        return int_to_chain(quot - 1, base) + letter

class OutOfChainsError(Exception):
    pass

def rename_chains(structure):
    """
    Renames chains to be one-letter valid PDB chains.

    Existing one-letter chains are kept. Others are renamed uniquely.
    Raises OutOfChainsError if more than 62 chains are present.
    Returns a map between new and old chain IDs.
    """
    next_chain = 0
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}
    
    for o in structure.get_chains():
        if len(o.id) != 1:
            if o.id[0] not in chainmap:
                chainmap[o.id[0]] = o.id
                o.id = o.id[0]
            else:
                c = int_to_chain(next_chain)
                while c in chainmap:
                    next_chain += 1
                    if next_chain >= 62:
                        raise OutOfChainsError()
                    c = int_to_chain(next_chain)
                chainmap[c] = o.id
                o.id = c
    return chainmap

def sanitize_residue_names(structure):
    """
    Truncates all residue names to 3 characters (PDB format limit).
    Logs a warning if truncation occurs.
    """
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.resname
                if len(resname) > 3:
                    truncated = resname[:3]
                    logging.warning(f"Truncating residue name '{resname}' to '{truncated}'")
                    residue.resname = truncated

def convert_cif_to_pdb(ciffile, pdbfile):
    """
    Convert a CIF file to PDB format, handling chain renaming and residue name truncation.

    Args:
        ciffile (str): Path to input CIF file
        pdbfile (str): Path to output PDB file

    Returns:
        bool: True if conversion succeeds, False otherwise
    """
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)

    strucid = ciffile[:4] if len(ciffile) > 4 else "1xxx"

    # Parse CIF file
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(strucid, ciffile)

    # Rename chains
    try:
        rename_chains(structure)
    except OutOfChainsError:
        logging.error("Too many chains to represent in PDB format")
        return False

    # Truncate long ligand or residue names
    sanitize_residue_names(structure)

    # Write to PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdbfile)
    return True

def convert_cif_files_to_pdb(results_dir, save_dir, af_dir=False, high_iptm=False, i_ptm_cutoff = 0.5):
    """
    Convert all .cif files in results_dir to .pdb format and save in save_dir
    
    Args:
        results_dir (str): Directory containing .cif files
        save_dir (str): Directory to save converted .pdb files
        af_dir (bool): If True, look for _model.cif_model.cif files instead of .cif
    """
    confidence_scores = []
    os.makedirs(save_dir, exist_ok=True)
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if af_dir:
                if file.endswith('_model.cif'):
                    cif_path = os.path.join(root, file)
                    pdb_path = os.path.join(save_dir, file.replace('.cif', '.pdb'))
                    print(f"Converting {cif_path}")
                    
                    if high_iptm:
                        confidence_file_summary = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('summary_confidences.json')][0]
                        confidence_file = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('_confidences.json') and not f.endswith('summary_confidences.json')][0]
                        with open(confidence_file_summary) as f:
                            confidence_data = json.load(f)
                            iptm = confidence_data['iptm']

                        with open(confidence_file, 'r') as f:
                            confidence_data = json.load(f)
                            plddt = np.mean(confidence_data['atom_plddts'])
                            
                        if iptm > i_ptm_cutoff:
                            print(f"Converting {cif_path}") 
                            print(f"iptm score: {iptm}")
                            print(f"pdb_path: {pdb_path}")
                            convert_cif_to_pdb(cif_path, pdb_path)
                            confidence_scores.append({'file': file, 'iptm': iptm, 'plddt': plddt})

                    else:  
                        print(f"Converting {cif_path}")
                        convert_cif_to_pdb(cif_path, pdb_path)
            else:
                if file.endswith('.cif'):
                    cif_path = os.path.join(root, file)
                    pdb_path = os.path.join(save_dir, file.replace('.cif', '.pdb'))
                    print(f"Converting {cif_path}")

                    if high_iptm:
                        confidence_files = [f for f in os.listdir(root) if f.startswith('confidence_') and f.endswith('.json')]
                        if confidence_files:
                            confidence_file = os.path.join(root, confidence_files[0])
                            with open(confidence_file) as f:
                                confidence_data = json.load(f)
                                iptm = confidence_data['iptm']
                            if iptm > i_ptm_cutoff:
                                print(f"Converting {cif_path}")
                                print(f"Confidence file: {confidence_file}")
                                print(f"iptm score: {iptm}")
                                convert_cif_to_pdb(cif_path, pdb_path)
                        
                    else:
                        print(f"Converting {cif_path}")
                        convert_cif_to_pdb(cif_path, pdb_path)
                        
            if confidence_scores:
                confidence_scores_path = os.path.join(save_dir, 'high_iptm_confidence_scores.csv')
                pd.DataFrame(confidence_scores).to_csv(confidence_scores_path, index=False)
                print(f"Saved confidence scores to {confidence_scores_path}")


def get_protein_ligand_interface(pdb_id, cutoff=6, non_protein_target=True, binder_chain='A', target_chain='B'):
    """
    Get interface residues between a protein and ligand/target from a PDB structure.
    
    Args:
        pdb_id (str): Path to PDB file
        cutoff (float): Distance cutoff in Angstroms for interface residues
        non_protein_target (bool): Whether to select non-protein ligand (True) or protein target chain (False)
        binder_chain (str): Chain ID of the protein binder chain
        target_chain (str): Chain ID of the target protein chain (if non_protein_target=False)
    
    Returns:
        list: Residue indices that are at the interface
    """
    # Load PDB structure
    pdb = parsePDB(pdb_id)
    
    # Get binder protein coordinates
    protein = pdb.select(f'protein and chain {binder_chain} and name CA')
    protein_coords = protein.getCoords()
    
    # Get ligand/target coordinates
    if non_protein_target:
        ligand = pdb.select('not protein and not water')
    else:
        ligand = pdb.select(f'protein and chain {target_chain} and name CA')
        
    ligand_coords = ligand.getCoords()
    
    # Calculate pairwise distances and find interface residues
    distances = np.sqrt(np.sum((protein_coords[:,None,:] - ligand_coords[None,:,:])**2, axis=-1))
    interacting = distances < cutoff
    interface_residues = list(np.nonzero(np.sum(interacting, axis=-1))[0])
    
    return interface_residues

def get_protein_ligand_interface_all_atom(pdb_id, cutoff=6, non_protein_target=True, binder_chain='A', target_chains="all"):
    print("target_chains", target_chains)
    """
    Get interface residues between a protein and ligand/target from a PDB structure.
    Args:
        pdb_id (str): Path to PDB file
        cutoff (float): Distance cutoff in Angstroms for interface residues
        non_protein_target (bool): Whether to select non-protein ligand (True) or protein target chain (False)
        binder_chain (str): Chain ID of the protein binder chain
        target_chains (list): Chain IDs of the target protein chains (if non_protein_target=False). If None, all chains except binder_chain are used.
    
    Returns:
        list: Residue indices that are at the interface
    """
    # Load PDB structure
    pdb = parsePDB(pdb_id)
    
    # Get binder protein coordinates
    protein = pdb.select(f'protein and chain {binder_chain}')
    protein_residues = set(protein.getResnums())
    

    if non_protein_target:
        ligand = pdb.select('not protein and not water')
        ligand_atoms = ligand.getCoords()
        protein_atoms = protein.getCoords()
        
        distances = np.sqrt(np.sum((protein_atoms[:,None,:] - ligand_atoms[None,:,:])**2, axis=-1))
        interacting = distances < cutoff
        
        interface_residues = set()
        for i in range(len(protein)):
            if np.any(interacting[i]):
                interface_residues.add(protein.getResnums()[i])
                
    else:
        if target_chains == "all":
            all_chains = set([chain.getChid() for chain in pdb.select('protein').getHierView().iterChains()])
            target_chains = list(all_chains - {binder_chain})
            print("target_chains", target_chains)
            
        interface_residues = set()
        for target_chain in target_chains:
            ligand = pdb.select(f'protein and chain {target_chain}')
            ligand_atoms = ligand.getCoords()
            protein_atoms = protein.getCoords()
            
            # Calculate all atom distances
            distances = np.sqrt(np.sum((protein_atoms[:,None,:] - ligand_atoms[None,:,:])**2, axis=-1))
            interacting = distances < cutoff
            
            # Map interacting atoms back to residues
            for i in range(len(protein)):
                if np.any(interacting[i]):
                    interface_residues.add(protein.getResnums()[i])
    
    sorted_residues = sorted(list(protein_residues))
    interface_indices = [sorted_residues.index(res) for res in interface_residues]
    
    return sorted(interface_indices)

def run_ligandmpnn_redesign(
    base_dir,
    pdb_dir,
    boltz_path,
    yaml_dir,
    ligandmpnn_config,
    top_k=5,
    cutoff=6,
    non_protein_target=True,
    binder_chain='A',
    target_chains=['B'],
    fix_interface=True,
    out_dir=None,
    lmpnn_yaml_dir=None,
    results_final_dir=None
):
    # Set default output directories if not provided
    if out_dir is None:
        out_dir = os.path.join(base_dir, '01_lmpnn_redesigned_fa')
    if lmpnn_yaml_dir is None:
        lmpnn_yaml_dir = os.path.join(base_dir, '01_lmpnn_redesigned_yaml')
    if results_final_dir is None:
        results_final_dir = os.path.join(base_dir, '01_lmpnn_redesigned')

    # Create required directories
    for directory in [out_dir, lmpnn_yaml_dir, results_final_dir]:
        os.makedirs(directory, exist_ok=True)

    # Collect all PDB paths for multi-state design
    pdb_paths = sorted([os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    if not pdb_paths:
        print(f"No PDB files found in {pdb_dir}")
        return

    # Use the first PDB name as the base name for outputs
    base_pdb_name = os.path.basename(pdb_paths[0]).split('.pdb')[0]
    
    # Calculate interface residues for the first structure (approximation for ensemble)
    # Ideally, one might want the union of interfaces, but using the first is a common heuristic
    interface_residues = get_protein_ligand_interface_all_atom(
        pdb_paths[0], cutoff=cutoff, non_protein_target=non_protein_target, 
        binder_chain=binder_chain, target_chains=target_chains
    )
    print(f"len interface_residues (from {base_pdb_name}): {len(interface_residues)}")

    with open(ligandmpnn_config, 'r') as f:
        config_dict = yaml.safe_load(f)

    if non_protein_target:
        model_type = "ligand_mpnn"
    else:
        model_type = "soluble_mpnn"

    config = SimpleNamespace(**config_dict)
    config.model_type = model_type
    config.seed = 111
    # Pass list of PDBs for multi-state
    config.pdb_path_multi = pdb_paths 
    config.out_folder = out_dir
    
    if fix_interface:
        config.fixed_residues = " ".join([f'{binder_chain}{item+1}' for item in interface_residues])
    
    config.batch_size = 16
    config.save_stats = 0
    config.chains_to_design = binder_chain
    
    # Run LigandMPNN on the ensemble
    output = main(config)
    
    # Process output FASTA
    # Note: LigandMPNN multi-state output naming might differ, assuming standard naming or using first pdb name
    # Usually it generates seqs/{pdb_name}.fa. If multi, check documentation/output. 
    # Assuming it uses the name of the first pdb in the list or a consolidated name.
    # For safety here, we look for the file that matches the base_pdb_name.
    fasta_path = os.path.join(out_dir, 'seqs', f'{base_pdb_name}.fa')
    
    if not os.path.exists(fasta_path):
         # If exact match fails, try finding any .fa file in the output seqs folder
         possible_fastas = list(Path(os.path.join(out_dir, 'seqs')).glob('*.fa'))
         if possible_fastas:
             fasta_path = str(possible_fastas[0])
         else:
             print(f"Error: No output fasta found at {fasta_path}")
             return

    print(fasta_path)
    # Read the existing fa file
    with open(fasta_path, 'r') as f:
        lines = f.readlines()
        sequences = []
        sequence_found = False  # Flag to check if sequence is found
        for line in lines[2:]:
            if line.startswith('>'):
                try:
                    overall_confidence = float(line.split(',')[4].split('=')[1])
                    ligand_confidence = line.split(',')[5].split('=')[1]
                    sequences.append((overall_confidence, ligand_confidence, ""))  # Store confidence and ligand
                    sequence_found = True  # Set flag to true if sequence is found
                except IndexError:
                     # Handle cases where header format might be slightly different
                     continue
            elif sequence_found:  # Check for the sequence line after finding confidence
                sequences[-1] = (sequences[-1][0], sequences[-1][1], line.strip())  # Add the corresponding sequence
                sequence_found = False  # Reset flag after capturing the sequence
    
    top_sequences = sorted(sequences, key=lambda x: x[0], reverse=True)[:top_k]
    
    for idx, (overall_confidence, ligand_confidence, sequence) in enumerate(top_sequences):
        matching_yamls = list(Path(yaml_dir).glob(f'{base_pdb_name.split("_results")[0]}*.yaml'))
        
        if matching_yamls:
            yaml_path = str(matching_yamls[0])  # Take the first matching yaml file
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Remove constraints
            yaml_data.pop('constraints', None)
            
            if not non_protein_target:
                binder_idx = chain_to_number[binder_chain]
                # Check bounds
                if binder_idx < len(yaml_data['sequences']):
                     yaml_data['sequences'][binder_idx]['protein']['sequence'] = sequence.split(':')[chain_to_number[binder_chain]]
            else:
                binder_idx = chain_to_number[binder_chain]
                if binder_idx < len(yaml_data['sequences']):
                    yaml_data['sequences'][binder_idx]['protein']['sequence'] = sequence

            # Replace .npz with .a3m in msa paths
            for seq in yaml_data['sequences']:
                if 'protein' in seq and 'msa' in seq['protein']:
                    msa_path = seq['protein']['msa']
                    if isinstance(msa_path, str) and msa_path.endswith('.npz'):
                        seq['protein']['msa'] = msa_path.replace('.npz', '.a3m')

            final_yaml_path = os.path.join(lmpnn_yaml_dir, f'{base_pdb_name}_{idx+1}.yaml')
            with open(final_yaml_path, 'w') as f:
                yaml.dump(yaml_data, f)

            import subprocess
            subprocess.run([boltz_path, 'predict', str(final_yaml_path), '--out_dir', str(results_final_dir), '--write_full_pae'])
            print(f"Completed processing {base_pdb_name} for sequence {idx+1}")


