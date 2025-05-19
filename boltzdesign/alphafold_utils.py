import os
from pathlib import Path
import yaml
def build_json_sequence(name, protein=None, protein_id=None, modification_ls=None, modification_chain=None, 
                       ligand=None, ligand_id=None, metal=None, metal_id=None, model_seeds=[1], dialect="alphafold3", version=1):
    """
    Build a customizable JSON structure for AlphaFold input with protein, ligand, and metal entries.

    Args:
        name (str): Name of the structure
        protein (list): List of protein sequences
        protein_id (list): List of protein chain IDs
        modification_ls (list): List of modifications to apply
        modification_chain (str): Chain ID to apply modifications to
        ligand (list): List of ligand SMILES strings
        ligand_id (list): List of ligand chain IDs
        metal (list): List of metal CCD codes
        metal_id (list): List of metal chain IDs
        model_seeds (list): List of model seed numbers
        dialect (str): Dialect of the JSON model
        version (int): Version of the JSON model

    Returns:
        dict: JSON structure for AlphaFold input
    """
    json_structure = {
        "name": name,
        "sequences": [],
        "modelSeeds": model_seeds,
        "dialect": dialect,
        "version": version
    }

    # Add protein information
    if protein and protein_id:
        for i, (seq, chain_id) in enumerate(zip(protein, protein_id)):
            protein_entry = {
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                    "pairedMsa": None,
                    "unpairedMsa": None,
                    "templates": []
                }
            }

            # Add modifications if this is the target chain
            if chain_id == modification_chain and modification_ls:
                protein_entry["protein"]["modifications"] = []
                for ptm_type, ptm_pos in modification_ls:
                    protein_entry["protein"]["modifications"].append({
                        "ptmType": ptm_type,
                        "ptmPosition": ptm_pos
                    })
            else:
                # Add default MSA for non-modified chains
                protein_entry["protein"]["pairedMsa"] = f">query\n{seq}"
                protein_entry["protein"]["unpairedMsa"] = f">query\n{seq}"

            json_structure["sequences"].append(protein_entry)

    # Add ligand information
    if ligand and ligand_id:
        for smiles, chain_id in zip(ligand, ligand_id):
            json_structure["sequences"].append({
                "ligand": {
                    "id": chain_id,
                    "smiles": smiles
                }
            })

    # Add metal information
    if metal and metal_id:
        for ccd_code, chain_id in zip(metal, metal_id):
            json_structure["sequences"].append({
                "ligand": {
                    "id": chain_id,
                    "ccdCodes": [ccd_code]
                }
            })

    return json_structure


def extract_sequences_and_format(a3m_file_path,replace_query_seq=False, query_seq=''):
    sequences = []
    first_sequence_reformatted = replace_query_seq  # Track if we have renamed the first sequence
    
    with open(a3m_file_path, 'r') as file:
        seq_id = None
        seq_data = ''
        
        for line in file:
            line = line.strip()
            
            # Skip comment lines or empty lines
            if line.startswith('#') or not line:
                continue
            
            # If the line starts with '>', it's a new sequence header
            if line.startswith('>'):
                # If we already have a sequence stored, save it
                if seq_id:
                    # Rename the first sequence header to 'wild-type' if replacing query sequence
                    if replace_query_seq and not first_sequence_reformatted:
                        seq_id = "wild-type"
                        first_sequence_reformatted = True
                    sequences.append(f">{seq_id}\n{seq_data}")
                
                # Update the sequence ID and reset the sequence data
                seq_id = line[1:]  # remove '>'
                seq_data = ''
            else:
                # Accumulate the sequence data
                seq_data += line
        
        # Add the last sequence to the list
        if seq_id:
            if replace_query_seq and not first_sequence_reformatted:  # Handle the case where the last sequence is the first
                seq_id = "wild-type"
            sequences.append(f">{seq_id}\n{seq_data}")
    
    if replace_query_seq:
        # Add the query sequence as the first sequence in the list
        query_seq_formatted = f">query\n{query_seq}"
        sequences.insert(0, query_seq_formatted)
    
    # Join all sequences into a single formatted string
    joined_string = "\n".join(sequences)
    return joined_string


def build_json_sequence_only_nucleotide_with_rna_msa(name, protein=None, protein_id=None, rna=None,  dna=None, ligand=None,model_seeds=[1], dialect="alphafold3", version=1):
    """
    Build a customizable JSON structure with protein, RNA, and ligand entries.

    :param name: Name of the structure
    :param protein: Protein sequence string
    :param rna: RNA sequence string 
    :param ligand: Ligand SMILES string
    :param model_seeds: List of model seed numbers (default is [1]).
    :param dialect: Dialect of the JSON model (default is 'alphafold3').
    :param version: Version of the JSON model (default is 1).
    :return: JSON structure as a dictionary.
    """
    json_structure = {
        "name": name,
        "sequences": [],
        "modelSeeds": model_seeds,
        "dialect": dialect,
        "version": version
    }

    # Add protein information if provided
    if protein:
        protein_entry = {
            "protein": {
                "id": protein_id,
                "sequence": protein,
                "pairedMsa": ">query\n"+protein,
                "unpairedMsa": ">query\n"+protein,
                "templates": []
            }
        }
        json_structure["sequences"].append(protein_entry)
    # Add RNA information if provided
    if rna:
        for item in rna:
            rna_structure = {
                "rna": {
                    "id": item['id'],
                    "sequence": item['sequence'],
                    "unpairedMsa": ">query\n"+item['sequence']
                }
            }
            json_structure["sequences"].append(rna_structure)

    if dna:
        ## no uniapredMSA for dna
        for item in dna:
            dna_structure = {
                "dna": {
                    "id": item['id'],
                    "sequence": item['sequence'],
                    # "unpairedMsa": ">query\n"+item['sequence']
                }
            }
            json_structure["sequences"].append(dna_structure)
    # Add ligand information if provided
    if ligand:
        for item in ligand:
            ligand_entry = {
                "ligand": {
                    "id": item['id'],
                    "smiles": item['smiles']
                }
            }
        json_structure["sequences"].append(ligand_entry)

    return json_structure
# def build_json_sequence_only(name, protein=None, protein_id=None, rna=None, rna_id=None, dna=None, dna_id=None, ligand=None, ligand_id=None, metal=None, metal_id=None, model_seeds=[1], dialect="alphafold3", version=1):
#     """
#     Build a customizable JSON structure with protein, RNA, and ligand entries.

#     :param name: Name of the structure
#     :param protein: Protein sequence string
#     :param rna: RNA sequence string 
#     :param ligand: Ligand SMILES string
#     :param model_seeds: List of model seed numbers (default is [1]).
#     :param dialect: Dialect of the JSON model (default is 'alphafold3').
#     :param version: Version of the JSON model (default is 1).
#     :return: JSON structure as a dictionary.
#     """
#     json_structure = {
#         "name": name,
#         "sequences": [],
#         "modelSeeds": model_seeds,
#         "dialect": dialect,
#         "version": version
#     }

#     # Add protein information if provided
#     if protein:
#         protein_entry = {
#             "protein": {
#                 "id": protein_id,
#                 "sequence": protein,
#                 "pairedMsa": ">query\n"+protein,
#                 "unpairedMsa": ">query\n"+protein,
#                 "templates": []
#             }
#         }
#         json_structure["sequences"].append(protein_entry)

#     # Add RNA information if provided
#     if rna:
#         rna_structure = {
#             "rna": {
#                 "id": rna_id,
#                 "sequence": rna
#             }
#         }
#         json_structure["sequences"].append(rna_structure)

#     if dna:
#         dna_structure = {
#             "dna": {
#                 "id": dna_id,
#                 "sequence": dna
#             }
#         }
#         json_structure["sequences"].append(dna_structure)
#     # Add ligand information if provided
#     if ligand:
#         ligand_entry = {
#             "ligand": {
#                 "id": ligand_id,
#                 "smiles": ligand
#             }
#         }
#         json_structure["sequences"].append(ligand_entry)
    
#     if metal:
#         metal_entry = {
#             "ligand": {
#                 "id": metal_id,
#                 "ccdCodes": [metal]
#             }
#         }
#         json_structure["sequences"].append(metal_entry)

#     return json_structure


