import os
import pyrosetta as pr
from scipy.spatial import cKDTree
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from Bio.PDB import (PDBParser, DSSP, Selection, Polypeptide, PDBIO,
                    Select, Chain, Superimposer, MMCIFParser)
from Bio.PDB.Selection import unfold_entities
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import re
import time
import argparse
import zipfile
from boltzdesign_utils import *

# Initialize PyRosetta with all needed options
dalphaball_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DAlphaBall.gcc')
pr.init(f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 1')

def radius_of_gyration(path, chain_id='B'):
    if path.endswith('.cif'):
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    structure = parser.get_structure('structure', path)
    model = structure[0]
    chain = model[chain_id]
    ca_coords = torch.tensor([atom.get_coord() for residue in chain for atom in residue if atom.get_name() == 'CA'])
    rg = torch.sqrt(torch.square(ca_coords - ca_coords.mean(0)).sum(-1).mean() + 1e-8)
    return rg.item(), len(ca_coords)

# Get chain B sequence from CIF file
from Bio.PDB import *

def get_sequence(cif_file, chain_id= 'B'):
    parser = MMCIFParser()
    structure = parser.get_structure('protein', cif_file)
    chain_b = structure[0][chain_id]
    sequence = ''.join([residue.resname for residue in chain_b])
    # Convert three letter codes to one letter codes
    from Bio.SeqUtils import seq1
    one_letter_seq = seq1(sequence)
    return one_letter_seq


def clean_pdb(pdb_file):
    # Read the pdb file and filter relevant lines
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [line for line in f_in if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END'))]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)

def pr_relax(pdb_file, relaxed_pdb_path):
    if not os.path.exists(relaxed_pdb_path):
        relax_start_time = time.time()

        # Generate pose
        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        ### Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(True) # enable sidechain movement
        mmf.set_bb(True) # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
        mmf.set_jump(False) # disable whole chain movement

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf) # set MoveMap
        fastrelax.max_iter(200) # default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # Align relaxed structure to original trajectory
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # output relaxed and aligned PDB
        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)

        relax_time = time.time() - relax_start_time
        relax_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(relax_time // 3600), int((relax_time % 3600) // 60), int(relax_time % 60))}"
        #print("Relaxation took: "+relax_time_text)
        
def filter_data(main_path):
    filtered_pdb= []
    for pdb in os.listdir(main_path):
        try:
            pdb_path = os.path.join(main_path, pdb)
            data = np.load(f"{pdb_path}/pdb/best.npz")
            L=len(data['pae'])
            print(pdb, data['metrics'])
            if np.mean(data['plddt']) > 0.85:
                filtered_pdb.append(pdb_path)

        except:
            continue
            
    return filtered_pdb

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])
 
    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def score_interface(pdb_file, binder_chain="B"):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    iam.set_interface("A_B")
    scorefxn = pr.get_fa_scorefxn()
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(pdb_file, binder_chain)
    interface_residues_pdb_ids = []
    
    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ','.join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set('ACFILMPVWY')
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value # shape complementarity
    interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds
    interface_dG = iam.get_interface_dG() # interface dG
    interface_dSASA = iam.get_interface_delta_sasa() # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat() # interface pack stat score
    interface_dG_SASA_ratio = interfacescore.dG_dSASA_ratio * 100 # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter('<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true" dalphaball_sasa="1" probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />')
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100 # Hbonds per interface size percentage
        interface_bunsch_percentage = (interface_delta_unsat_hbonds / interface_nres) * 100 # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {pose.pdb_info().chain(pose.conformation().chain_begin(i)): p for i, p in zip(range(1, pose.num_chains()+1), pose.split_by_chain())}[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core = False, pick_boundary = False, pick_surface = True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0 
    
    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if res.is_apolar() == True or res.name() == 'PHE' or res.name() == 'TRP' or res.name() == 'TYR':
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count/total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
    'binder_score': binder_score,
    'surface_hydrophobicity': surface_hydrophobicity,
    'interface_sc': interface_sc,
    'interface_packstat': interface_packstat,
    'interface_dG': interface_dG,
    'interface_dSASA': interface_dSASA,
    'interface_dG_SASA_ratio': interface_dG_SASA_ratio,
    'interface_fraction': interface_binder_fraction,
    'interface_hydrophobicity': interface_hydrophobicity,
    'interface_nres': interface_nres,
    'interface_interface_hbonds': interface_interface_hbonds,
    'interface_hbond_percentage': interface_hbond_percentage,
    'interface_delta_unsat_hbonds': interface_delta_unsat_hbonds,
    'interface_delta_unsat_hbonds_percentage': interface_bunsch_percentage
    }

    # round to two decimal places
    interface_scores = {k: round(v, 2) if isinstance(v, float) else v for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str

# align pdbs to have same orientation
def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    # initiate poses
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    align = AlignChainMover()
    align.pose(reference_pose)

    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(',')[0]
    align_chain_id = align_chain_id.split(',')[0]

    # Get the chain number corresponding to the chain ID in the poses
    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(reference_chain_id, reference_pose)
    align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(align_chain_id, align_pose)

    align.source_chain(align_chain)
    align.target_chain(reference_chain)
    align.apply(align_pose)

    # Overwrite aligned pdb
    align_pose.dump_pdb(align_pdb)
    clean_pdb(align_pdb)

# calculate the rmsd without alignment
def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # Define chain selectors for the reference and align chains
    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    # Apply selectors to get residue subsets
    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    # Convert subsets to residue index vectors
    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)

    # Create empty subposes
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    # Fill subposes
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # Calculate RMSD using the RMSDMetric
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)

# Relax designed structure
def pr_relax(pdb_file, relaxed_pdb_path):
    if not os.path.exists(relaxed_pdb_path):
        # Generate pose
        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        ### Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(True) # enable sidechain movement
        mmf.set_bb(True) # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
        mmf.set_jump(False) # disable whole chain movement

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf) # set MoveMap
        fastrelax.max_iter(200) # default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # Align relaxed structure to original trajectory
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # Copy B factors from start_pose to pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # Get the B factor of the first heavy atom in the residue
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # output relaxed and aligned PDB
        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)


def get_binder_chain(pdb_file):
    parser = PDBParser(QUIET=True)
    binder_structure = parser.get_structure('protein', pdb_file)
    chain_ids = [chain.id for model in binder_structure for chain in model]
    return chain_ids[-1]

def measure_rosetta_energy(pdbs_path, pdbs_apo_path, save_dir, binder_holo_chain='B', binder_apo_chain='A'):
    # Create relaxed output directory 
    relaxed_dir = pdbs_path + '_relaxed'
    os.makedirs(relaxed_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(pdbs_path, 'rosetta_energy.csv')
    
    # Initialize DataFrame
    df = pd.DataFrame()
    
    # If file exists, load it to get processed files
    processed_files = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_files = set(existing_df['Model'].values)
    for pdb_file in os.listdir(pdbs_path):
        if pdb_file.endswith('.pdb') and not pdb_file.startswith('relax_'):
            # Skip if already processed
            if pdb_file in processed_files:
                continue

            try:
                design_pathway = os.path.join(pdbs_path, pdb_file)
                relax_pathway = os.path.join(relaxed_dir, f"relax_{pdb_file}")
                binder_chain = get_binder_chain(design_pathway)
                pr_relax(design_pathway, relax_pathway)
                trajectory_interface_scores, trajectory_interface_AA, trajectory_interface_residues = score_interface(relax_pathway, binder_chain)
                print(trajectory_interface_scores)
                
                row_data = {'PDB': relaxed_dir, 'Model':  f"relax_{pdb_file}"}
                row_data.update(trajectory_interface_scores)
                
                # Append new row to DataFrame
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                processed_files.add(pdb_file)
            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
    
    # Save all results at once
    if os.path.exists(output_path):
        # Append new results to existing file
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        df.to_csv(output_path, index=False)
    
    print(f"Results saved to: {output_path}")


    all_filtered_rows = []
    all_failed_rows = []
    success_sample_num =0

    
    mask = (
        (df['binder_score'] < 0) &
        (df['surface_hydrophobicity'] < 0.35) &
        (df['interface_sc'] > 0.55) &
        (df['interface_packstat'] > 0) &
        (df['interface_dG'] < 0) &
        (df['interface_dSASA'] > 1) &
        (df['interface_dG_SASA_ratio'] < 0) &
        (df['interface_nres'] > 7) &
        (df['interface_interface_hbonds'] > 3) &
        (df['interface_hbond_percentage'] > 0) &
        (df['interface_delta_unsat_hbonds'] < 4)
    )
    filtered_df = df[mask].copy()
    failed_df = []
    
    print(f"Number of designs passing all filters: {len(filtered_df)}")
    print(f"Number of failed designs: {len(failed_df)}")

    if len(filtered_df) > 0:
        for i in range(len(filtered_df)):
            try:
                row = filtered_df.iloc[i].copy()  # Create a copy to avoid SettingWithCopyWarning
                cif_path = row['PDB'] + '/' + row['Model']

                rg, length = radius_of_gyration(cif_path, chain_id=binder_holo_chain)
                
                # Extract model name without relax_ prefix if present
                model_base = row['Model'].split('relax_')[-1].split('_model.pdb')[0] if row['Model'].startswith('relax') else row['Model'].split('_model.pdb')[0]
                
                # Construct paths
                base_path = '/'.join(row['PDB'].split('/')[:-1]) + '/02_design_final_af3/' + model_base
                confidenece_json_1 = f"{base_path}/{model_base}_summary_confidences.json"
                confidenece_json_2 = f"{base_path}/{model_base}_confidences.json"
                af_cif = f"{base_path}/{model_base}_model.cif"
                
                aa_seq = get_sequence(af_cif, chain_id=binder_holo_chain)
                
                # Set PDB paths
                if row['Model'].startswith('relax'):
                    af_holo_pdb = pdbs_path + '/' + row['Model'].split('relax_')[1]
                    af_apo_pdb = pdbs_apo_path + '/' + row['Model'].split('relax_')[1]
                else:
                    af_holo_pdb = pdbs_path + '/' + row['Model']
                    af_apo_pdb = pdbs_apo_path + '/' + row['Model']

                xyz_holo, seq_holo = get_CA_and_sequence(af_holo_pdb, chain_id=binder_holo_chain)
                xyz_apo, seq_apo = get_CA_and_sequence(af_apo_pdb, chain_id=binder_apo_chain)
                rmsd = np_rmsd(xyz_holo, xyz_apo)
                row['apo_holo_rmsd'] = rmsd
            
                with open(confidenece_json_1, 'r') as f:
                    confidence_data = json.load(f)
                    row['iptm'] = confidence_data['iptm']
                    
                with open(confidenece_json_2, 'r') as f:
                    confidence_data = json.load(f)
                    
                    row['plddt'] = np.mean(confidence_data['atom_plddts'])
                    pae_matrix= np.array(confidence_data['pae'])
                    protein_len = len(aa_seq)
                    interface_pae1 = np.mean(pae_matrix[:protein_len, protein_len:])
                    interface_pae2 = np.mean(pae_matrix[protein_len:, :protein_len])
                    i_pae = (interface_pae1 + interface_pae2) / 2

                row['i_pae'] = i_pae
                row['rg'] = rg
                row['aa_seq'] = aa_seq

                print(row['iptm'], row['plddt'], rg, row['i_pae'], row['apo_holo_rmsd'])
                if row['iptm'] > 0.5 and row['plddt'] > 80 and rg < 17 and row['i_pae'] < 15 and row['apo_holo_rmsd'] < 3.5:
                    shutil.copy(Path(row['PDB']) / row['Model'], save_dir + '/' + row['Model'])
                    all_filtered_rows.append(row)
                    success_sample_num+=1
                else:
                    all_failed_rows.append(row)
            except Exception as e:
                print(f"Error processing {row['Model']}: {e}")
                continue

    # Add all failed cases
    print("success_sample_num", success_sample_num)
    for i in range(len(failed_df)):
        all_failed_rows.append(failed_df.iloc[i])

    success_csv = os.path.join(save_dir, 'success_designs.csv')
    failed_csv = os.path.join(save_dir, 'failed_designs.csv')
    zip_path = save_dir + '.zip'
    
    save_df = pd.DataFrame(all_filtered_rows)
    save_df.to_csv(success_csv, index=False)
    
    print("Number of Success designs", len(save_df))

    failed_df = pd.DataFrame(all_failed_rows)
    failed_df.to_csv(failed_csv, index=False)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, save_dir)
                zipf.write(file_path, arcname)
    
