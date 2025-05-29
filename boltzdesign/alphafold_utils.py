import os
from pathlib import Path
import yaml
import json

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
}


def process_yaml_files(yaml_dir, af_input_dir, af_input_apo_dir, target_type='small_molecule', binder_chain='A', target_chain='B'):
    """
    Process YAML files and generate JSON files for different binder types.
    
    Args:
        yaml_dir (Path): Directory containing YAML files
        af_input_dir (str): Output directory for complex JSON files
        af_input_apo_dir (str): Output directory for apo JSON files
        target_type (str): Type of binder ('small', 'ppi', 'na', 'metal')
    """
    
    for yaml_path in Path(yaml_dir).glob('*.yaml'):
        name = os.path.basename(yaml_path).split('.yaml')[0]
        
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

            
        if target_type == 'small_molecule':
            # Small molecule binder
            binder_seq = yaml_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence']
            other_seqs = [seq for i, seq in enumerate(yaml_data['sequences']) if i != chain_to_number[binder_chain]]
            ligand_smiles = []
            ligand_chain = []
            for seq in other_seqs:
                if 'ligand' in seq:
                    ligand_smiles.append(seq['ligand']['smiles'])
                    ligand_chain.append(seq['ligand']['id'])

            json_result = build_json_sequence(name, 
                protein=[binder_seq],
                protein_id=[binder_chain],
                ligand=ligand_smiles,
                ligand_id=ligand_chain) 

            json_result_apo = build_json_sequence(name,
                protein=[binder_seq],
                protein_id=[binder_chain])  
            
            
        elif target_type == 'protein':
            # Protein-protein binder
            target_seqs = []
            target_msas = []
            target_chains = []
            modification_ls = []
            modification_chain = []
            
            binder_seq = yaml_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence']
            
            # Get all target sequences and MSAs
            for i, seq in enumerate(yaml_data['sequences']):
                if i != chain_to_number[binder_chain] and 'protein' in seq:
        
                    target_seqs.append(seq['protein']['sequence'])
                    target_msas.append(seq['protein']['msa'].replace('experiment_candidates','inputs'))
                    target_chains.append(chr(ord('A') + len(target_seqs) - 1))  # A, B, C etc
                    if 'modifications' in seq['protein']:
                        modification_ls.append(seq['protein']['modifications'])
                        modification_chain.append(seq['protein']['id'][0])
                    else:
                        modification_ls.append([])
                        modification_chain.append([])
                    
            # Process MSAs for each target
            processed_msas = []
            for target_seq, msa_path in zip(target_seqs, target_msas):
                
                if msa_path != 'empty':
                    protein_msa = extract_sequences_and_format(msa_path,
                        replace_query_seq=True, 
                        query_seq=target_seq)
                    processed_msas.append(protein_msa)
                else:
                    processed_msas.append(f">query\n{target_seq}")
                
            # Add binder sequence and chain
            all_seqs = target_seqs + [binder_seq]
            all_chains = target_chains + [chr(ord('A') + len(target_seqs))]
            
            json_result = build_json_sequence(name,
                protein=all_seqs,
                protein_id=all_chains,
                modification_ls=modification_ls,
                modification_chain=modification_chain)
                
            # Add MSAs for target chains
            for i, msa in enumerate(processed_msas):
                json_result['sequences'][i]['protein']['pairedMsa'] = msa
                json_result['sequences'][i]['protein']['unpairedMsa'] = msa
                
            
            json_result_apo = build_json_sequence(name,
                protein=[binder_seq],
                protein_id=['A'])
                
        elif target_type == 'dna' or target_type == 'rna':
            # Nucleic acid binder
            protein_seq = yaml_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence']
            
            # Process RNA/DNA sequences
            rna_seqs = []
            dna_seqs = []
            other_seqs = [seq for i, seq in enumerate(yaml_data['sequences']) if i != chain_to_number[binder_chain]]
            for seq in other_seqs:
                if 'rna' in seq:
                    rna_seqs.append({
                        'id': seq['rna']['id'][0],
                        'sequence': seq['rna']['sequence']
                    })
                if 'dna' in seq:
                    dna_seqs.append({
                        'id': seq['dna']['id'][0],
                        'sequence': seq['dna']['sequence']
                    })
                    
            rna_seqs = None if len(rna_seqs) == 0 else rna_seqs
            dna_seqs = None if len(dna_seqs) == 0 else dna_seqs
            
            json_result = build_json_sequence(name,
                protein=protein_seq,
                protein_id='A',
                rna=rna_seqs,
                dna=dna_seqs)
                
            json_result_apo = build_json_sequence(name,
                protein=protein_seq,
                protein_id='A')
                
        elif target_type == 'metal':
            # Metal binder
            binder_seq = yaml_data['sequences'][chain_to_number[binder_chain]]['protein']['sequence']
            
            # Process metal sequences
            metal_seqs = []
            metal_chains = []
            other_seqs = [seq for i, seq in enumerate(yaml_data['sequences']) if i != chain_to_number[binder_chain]]
            for seq in other_seqs:
                if 'ligand' in seq and 'ccd' in seq['ligand']:
                    metal_seqs.append(seq['ligand']['ccd'])
                    metal_chains.append(seq['ligand']['id'])
            
            json_result = build_json_sequence(name,
                protein=[binder_seq],
                protein_id=[binder_chain],
                metal=metal_seqs,
                metal_id=metal_chains)
                
            json_result_apo = build_json_sequence(name,
                protein=[binder_seq],
                protein_id=[binder_chain])
                
        # Write output files
        with open(os.path.join(af_input_dir, f'{name}.json'), 'w') as f:
            json.dump(json_result, f, indent=4)
            
        with open(os.path.join(af_input_apo_dir, f'{name}.json'), 'w') as f:
            json.dump(json_result_apo, f, indent=4)

def build_json_sequence(name, protein=None, protein_id=None, modification_ls=None, modification_chain=None,
                       ligand=None, ligand_id=None, metal=None, metal_id=None, rna=None, dna=None,
                       model_seeds=[1], dialect="alphafold3", version=1):
    """
    Build a customizable JSON structure for AlphaFold input with protein, ligand, metal, RNA and DNA entries.

    Args:
        name (str): Name of the structure
        protein (list or str): List of protein sequences or single sequence
        protein_id (list or str): List of protein chain IDs or single ID
        modification_ls (list): List of modifications to apply
        modification_chain (str): Chain ID to apply modifications to
        ligand (list): List of ligand SMILES strings
        ligand_id (list): List of ligand chain IDs
        metal (list): List of metal CCD codes
        metal_id (list): List of metal chain IDs
        rna (list): List of RNA sequences with format [{'id': chain_id, 'sequence': seq}]
        dna (list): List of DNA sequences with format [{'id': chain_id, 'sequence': seq}]
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

    # Convert single protein/id to list for consistent handling
    if protein and not isinstance(protein, list):
        protein = [protein]
    if protein_id and not isinstance(protein_id, list):
        protein_id = [protein_id]

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
            if modification_chain is not None and modification_ls is not None:
                if chain_id in modification_chain and modification_ls[modification_chain.index(chain_id)]:
                    protein_entry["protein"]["modifications"] = []
                    for item in modification_ls[i]:
                        protein_entry["protein"]["modifications"].append({
                            "ptmType": item['ccd'],
                            "ptmPosition": item['position']
                        })
            else:
                # Add default MSA for non-modified chains
                protein_entry["protein"]["pairedMsa"] = f">query\n{seq}"
                protein_entry["protein"]["unpairedMsa"] = f">query\n{seq}"

            json_structure["sequences"].append(protein_entry)

    # Add RNA information
    if rna:
        for item in rna:
            rna_structure = {
                "rna": {
                    "id": item['id'],
                    "sequence": item['sequence'],
                    "unpairedMsa": f">query\n{item['sequence']}"
                }
            }
            json_structure["sequences"].append(rna_structure)

    # Add DNA information
    if dna:
        for item in dna:
            dna_structure = {
                "dna": {
                    "id": item['id'],
                    "sequence": item['sequence']
                }
            }
            json_structure["sequences"].append(dna_structure)

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

# def build_json_sequence(name, protein=None, protein_id=None, modification_ls=None, modification_chain=None, 
#                        ligand=None, ligand_id=None, metal=None, metal_id=None, model_seeds=[1], dialect="alphafold3", version=1):
#     """
#     Build a customizable JSON structure for AlphaFold input with protein, ligand, and metal entries.

#     Args:
#         name (str): Name of the structure
#         protein (list): List of protein sequences
#         protein_id (list): List of protein chain IDs
#         modification_ls (list): List of modifications to apply
#         modification_chain (str): Chain ID to apply modifications to
#         ligand (list): List of ligand SMILES strings
#         ligand_id (list): List of ligand chain IDs
#         metal (list): List of metal CCD codes
#         metal_id (list): List of metal chain IDs
#         model_seeds (list): List of model seed numbers
#         dialect (str): Dialect of the JSON model
#         version (int): Version of the JSON model

#     Returns:
#         dict: JSON structure for AlphaFold input
#     """
#     json_structure = {
#         "name": name,
#         "sequences": [],
#         "modelSeeds": model_seeds,
#         "dialect": dialect,
#         "version": version
#     }

#     # Add protein information
#     if protein and protein_id:
#         for i, (seq, chain_id) in enumerate(zip(protein, protein_id)):
#             protein_entry = {
#                 "protein": {
#                     "id": chain_id,
#                     "sequence": seq,
#                     "pairedMsa": None,
#                     "unpairedMsa": None,
#                     "templates": []
#                 }
#             }

#             # Add modifications if this is the target chain
#             if chain_id == modification_chain and modification_ls:
#                 protein_entry["protein"]["modifications"] = []
#                 for ptm_type, ptm_pos in modification_ls:
#                     protein_entry["protein"]["modifications"].append({
#                         "ptmType": ptm_type,
#                         "ptmPosition": ptm_pos
#                     })
#             else:
#                 # Add default MSA for non-modified chains
#                 protein_entry["protein"]["pairedMsa"] = f">query\n{seq}"
#                 protein_entry["protein"]["unpairedMsa"] = f">query\n{seq}"

#             json_structure["sequences"].append(protein_entry)

#     # Add ligand information
#     if ligand and ligand_id:
#         for smiles, chain_id in zip(ligand, ligand_id):
#             json_structure["sequences"].append({
#                 "ligand": {
#                     "id": chain_id,
#                     "smiles": smiles
#                 }
#             })

#     # Add metal information
#     if metal and metal_id:
#         for ccd_code, chain_id in zip(metal, metal_id):
#             json_structure["sequences"].append({
#                 "ligand": {
#                     "id": chain_id,
#                     "ccdCodes": [ccd_code]
#                 }
#             })

#     return json_structure


# def extract_sequences_and_format(a3m_file_path,replace_query_seq=False, query_seq=''):
#     sequences = []
#     first_sequence_reformatted = replace_query_seq  # Track if we have renamed the first sequence
    
#     with open(a3m_file_path, 'r') as file:
#         seq_id = None
#         seq_data = ''
        
#         for line in file:
#             line = line.strip()
            
#             # Skip comment lines or empty lines
#             if line.startswith('#') or not line:
#                 continue
            
#             # If the line starts with '>', it's a new sequence header
#             if line.startswith('>'):
#                 # If we already have a sequence stored, save it
#                 if seq_id:
#                     # Rename the first sequence header to 'wild-type' if replacing query sequence
#                     if replace_query_seq and not first_sequence_reformatted:
#                         seq_id = "wild-type"
#                         first_sequence_reformatted = True
#                     sequences.append(f">{seq_id}\n{seq_data}")
                
#                 # Update the sequence ID and reset the sequence data
#                 seq_id = line[1:]  # remove '>'
#                 seq_data = ''
#             else:
#                 # Accumulate the sequence data
#                 seq_data += line
        
#         # Add the last sequence to the list
#         if seq_id:
#             if replace_query_seq and not first_sequence_reformatted:  # Handle the case where the last sequence is the first
#                 seq_id = "wild-type"
#             sequences.append(f">{seq_id}\n{seq_data}")
    
#     if replace_query_seq:
#         # Add the query sequence as the first sequence in the list
#         query_seq_formatted = f">query\n{query_seq}"
#         sequences.insert(0, query_seq_formatted)
    
#     # Join all sequences into a single formatted string
#     joined_string = "\n".join(sequences)
#     return joined_string


# def build_json_sequence_only_nucleotide_with_rna_msa(name, protein=None, protein_id=None, rna=None,  dna=None, ligand=None,model_seeds=[1], dialect="alphafold3", version=1):
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
#         for item in rna:
#             rna_structure = {
#                 "rna": {
#                     "id": item['id'],
#                     "sequence": item['sequence'],
#                     "unpairedMsa": ">query\n"+item['sequence']
#                 }
#             }
#             json_structure["sequences"].append(rna_structure)

#     if dna:
#         ## no uniapredMSA for dna
#         for item in dna:
#             dna_structure = {
#                 "dna": {
#                     "id": item['id'],
#                     "sequence": item['sequence'],
#                     # "unpairedMsa": ">query\n"+item['sequence']
#                 }
#             }
#             json_structure["sequences"].append(dna_structure)
#     # Add ligand information if provided
#     if ligand:
#         for item in ligand:
#             ligand_entry = {
#                 "ligand": {
#                     "id": item['id'],
#                     "smiles": item['smiles']
#                 }
#             }
#         json_structure["sequences"].append(ligand_entry)

#     return json_structure
