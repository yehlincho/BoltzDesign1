import os
import numpy as np
import json
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser, PDBIO, PPBuilder, Select

# Standard amino acid 3-letter to 1-letter code mapping
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

def cif_to_pdb(cif_file, pdb_file, remove_cif=False):
    import gemmi
    st = gemmi.read_structure(str(cif_file))
    st.write_pdb(str(pdb_file))

    if remove_cif:
        os.remove(cif_file)


def pdb_to_cif(pdb_file, cif_file=None):
    """
    Convert a PDB file to mmCIF format.
    
    Args:
        pdb_file: Path to input PDB file
        cif_file: Path to output CIF file (default: same name with .cif extension)
    
    Returns:
        Path to the output CIF file
    """
    if cif_file is None:
        cif_file = pdb_file.rsplit('.', 1)[0] + '.cif'
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(cif_file))
    
    # Add required release date for AF3 templates
    add_release_date_to_cif(cif_file)
    
    return cif_file


def add_release_date_to_cif(cif_file, release_date="2020-01-01"):
    """
    Add the required _pdbx_audit_revision_history.revision_date field to a CIF file.
    AF3 requires this for template structures.
    
    Args:
        cif_file: Path to the CIF file to modify
        release_date: Date string in ISO-8601 format (YYYY-MM-DD)
    """
    with open(cif_file, 'r') as f:
        content = f.read()
    
    # Check if revision history already exists
    if '_pdbx_audit_revision_history.revision_date' in content:
        return
    
    # Add the required metadata block
    revision_block = f"""
#
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.data_content_type
_pdbx_audit_revision_history.major_revision
_pdbx_audit_revision_history.minor_revision
_pdbx_audit_revision_history.revision_date
1 'Structure model' 1 0 {release_date}
#
"""
    
    # Insert after the data_ line or at the beginning
    if content.startswith('data_'):
        # Find the end of the data_ line
        newline_pos = content.find('\n')
        if newline_pos != -1:
            content = content[:newline_pos+1] + revision_block + content[newline_pos+1:]
        else:
            content = content + revision_block
    else:
        content = revision_block + content
    
    with open(cif_file, 'w') as f:
        f.write(content)


def sequence_from_structure(structure_file, chain_ids=None):
    """
    Extract amino acid sequences from a PDB or CIF structure file.
    
    Args:
        structure_file: Path to PDB or CIF file
        chain_ids: List of chain IDs to extract (default: all chains)
    
    Returns:
        Dict mapping chain_id -> sequence string
    """
    # Determine parser based on file extension
    if structure_file.lower().endswith('.cif') or structure_file.lower().endswith('.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("structure", structure_file)
    
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_ids is not None and chain_id not in chain_ids:
                continue
            
            seq = []
            for residue in chain:
                # Skip hetero atoms (water, ligands, etc.)
                if residue.id[0] != ' ':
                    continue
                resname = residue.resname
                aa = AA_3TO1.get(resname, 'X')
                seq.append(aa)
            
            if seq:
                sequences[chain_id] = ''.join(seq)
    
    return sequences


class ChainSelect(Select):
    """Select a specific chain from a structure."""
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id


def extract_chain_to_cif(structure_file, chain_id, output_cif=None):
    """
    Extract a single chain from a structure and save as mmCIF.
    
    Args:
        structure_file: Path to PDB or CIF file
        chain_id: Chain ID to extract
        output_cif: Output CIF path (default: auto-generated)
    
    Returns:
        Path to the output single-chain CIF file
    """
    # Determine parser based on file extension
    if structure_file.lower().endswith('.cif') or structure_file.lower().endswith('.mmcif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    
    structure = parser.get_structure("structure", structure_file)
    
    # Generate output path if not specified
    if output_cif is None:
        base = structure_file.rsplit('.', 1)[0]
        output_cif = f"{base}_chain_{chain_id}.cif"
    
    # Save just the selected chain
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(output_cif), ChainSelect(chain_id))
    
    # Add required release date for AF3 templates
    add_release_date_to_cif(output_cif)
    
    return output_cif


def prepare_template(template_path, chain_ids=None, output_dir=None):
    """
    Prepare a template structure for AF3 input.
    - Converts PDB to CIF if needed
    - Extracts sequences from the structure
    
    Args:
        template_path: Path to template PDB or CIF file
        chain_ids: List of chain IDs to use (default: all protein chains)
        output_dir: Directory for converted CIF file (default: same as input)
    
    Returns:
        Tuple of (cif_path, sequences_dict)
    """
    # Convert PDB to CIF if needed
    if template_path.lower().endswith('.pdb'):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(template_path).rsplit('.', 1)[0]
            cif_path = os.path.join(output_dir, f"{base_name}.cif")
        else:
            cif_path = template_path.rsplit('.', 1)[0] + '.cif'
        
        print(f"Converting PDB to CIF: {template_path} -> {cif_path}")
        pdb_to_cif(template_path, cif_path)
    else:
        cif_path = template_path
    
    # Extract sequences
    sequences = sequence_from_structure(cif_path, chain_ids)
    
    return cif_path, sequences
    
def target_sequence_from_msa(msa_path):
    print("msa_path", msa_path)

    with open(msa_path, "r") as f:
        lines = f.readlines()

    header = None
    sequence = []

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Header line starts with '>'
        if line.startswith(">"):
            if header is None:
                # This is the first sequence header
                header = line[1:]  # Remove the '>' character
            else:
                # We've reached the second sequence, stop here
                break
        else:
            # This is a sequence line for the first sequence
            if header is not None:
                sequence.append(line)

    # Join all sequence lines and remove lowercase letters and insertions
    full_sequence = "".join(sequence)
    # Remove lowercase letters (insertions) to get the query sequence
    query_sequence = "".join(c for c in full_sequence if c.isupper() or c == "-")

    return query_sequence

def calculate_bias_schedule(bias_start, bias_end, cycle, num_cycles):
    
    # Parse start biases
    if isinstance(bias_start, str):
        start_dict = {}
        if bias_start:  # Handle empty string
            for item in bias_start.split(','):
                if ':' in item:
                    aa, val = item.split(':')
                    start_dict[aa.strip()] = float(val)
    else:
        start_dict = bias_start or {}
    
    # Parse end biases
    if isinstance(bias_end, str):
        end_dict = {}
        if bias_end:  # Handle empty string
            for item in bias_end.split(','):
                if ':' in item:
                    aa, val = item.split(':')
                    end_dict[aa.strip()] = float(val)
    else:
        end_dict = bias_end or {}
    
    # Get all amino acids mentioned in either start or end
    all_aas = set(start_dict.keys()) | set(end_dict.keys())
    
    # Calculate interpolated values
    current_biases = []
    for aa in sorted(all_aas):  # Sort for consistent ordering
        start_val = start_dict.get(aa, 0.0)  # Default to 0 if not specified
        end_val = end_dict.get(aa, 0.0)
        
        # Linear interpolation
        if num_cycles > 1:
            alpha = cycle / (num_cycles - 1)
            current_val = start_val + alpha * (end_val - start_val)
        else:
            current_val = start_val
        
        current_biases.append(f"{aa}:{current_val:.3f}")
    
    return ','.join(current_biases) if current_biases else ''


def compute_binder_iptm(full_pae: np.ndarray, asym_ids: np.ndarray, num_tokens: int) -> float:
    """Binder-specific iPTM: TM-score formula restricted to binder (asym_id=0) → target PAE pairs."""
    if full_pae.ndim == 2:
        full_pae = full_pae[np.newaxis, ...]
    pae   = full_pae[:, :num_tokens, :num_tokens]
    asym  = asym_ids[:num_tokens]
    binder_mask = asym == 0
    target_mask = asym != 0
    if not binder_mask.any() or not target_mask.any():
        return None
    d0 = max(1.24 * (num_tokens - 15) ** (1 / 3) - 1.8, 1.0)
    sub_pae = pae[:, binder_mask, :][:, :, target_mask]
    scores  = 1.0 / (1.0 + (sub_pae / d0) ** 2)
    return float(scores.max(axis=2).mean(axis=1).mean())


def build_data_dictionary(
    design_protein_length: int = 12,
    binder_chain: str = "A",
    binder_sequence: str = None,
    binder_type: str = "protein",  # Can be "protein", "dna", or "rna"
    # Target protein parameters (can be lists for multimers)
    protein_id: str | list[str] = "",
    protein_seq: str | list[str] = "",
    protein_msa: str | list[str] = "",
    # Ligand parameters (can be lists for multiple ligands)
    ligand_id: str | list[str] = "C",
    ligand_smiles: str | list[str] = "",
    ligand_ccd: str | list[str] = "",
    # Nucleic acid parameters (can be lists for multiple nucleic acids)
    nucleic_type: str | list[str] = "dna",
    nucleic_id: str | list[str] = "D",
    nucleic_seq: str | list[str] = "",
    # Template parameters
    template_path: str | list[str] = "",
    template_chain_id: str | list[str] = "",
    # Output directory for template CIF files
    output_dir: str = "",
    # Model parameters
    model_seed: int = None,
    # Sequence generation parameters
    percent_X: float = 100,
    # Binder (antibody) MSA — a3m text to attach to the binder chain. For antibodies
    # the framework has real homologs; matches AF3 default / [[project_af3_antibody_msa]].
    binder_msa: str = "",
) -> tuple[dict, str]:
    """
    Build an AlphaFold3-compatible input dictionary.
    Supports multimers by accepting lists for protein, ligand, and nucleic acid parameters.
    
    Args:
        binder_type: Type of binder molecule - "protein", "dna", or "rna"
    
    Returns:
        tuple: (data_dict, mode) where data_dict contains the folding input
               and mode is either "binder" or "unconditional"
    """
    import random
    # Helper function to ensure parameter is a list
    def make_list(param):
        if isinstance(param, list):
            return param
        if not param:
            return []
        # Split by comma or colon if string contains these delimiters
        if isinstance(param, str) and (',' in param or ':' in param):
            # Replace colons with commas for uniform splitting
            param = param.replace(':', ',')
            return [item.strip() for item in param.split(',') if item.strip()]
        return [param]
    
    # Convert all parameters to lists for uniform handling
    protein_ids = make_list(protein_id)
    protein_seqs = make_list(protein_seq)
    protein_msas = make_list(protein_msa)
    ligand_ids = make_list(ligand_id)
    ligand_smiles_list = make_list(ligand_smiles)
    ligand_ccds = make_list(ligand_ccd)
    nucleic_types = make_list(nucleic_type)
    nucleic_ids = make_list(nucleic_id)
    nucleic_seqs = make_list(nucleic_seq)
    template_paths = make_list(template_path)
    template_chain_ids = make_list(template_chain_id)
    
    # Pad lists to match lengths - now consider template_chain_ids as defining target count
    max_proteins = max(len(protein_ids), len(protein_seqs), len(protein_msas), len(template_chain_ids))
    protein_ids = protein_ids + [""] * (max_proteins - len(protein_ids))
    protein_seqs = protein_seqs + [""] * (max_proteins - len(protein_seqs))
    protein_msas = protein_msas + [""] * (max_proteins - len(protein_msas))
    template_paths = template_paths + [""] * (max_proteins - len(template_paths))
    template_chain_ids = template_chain_ids + [""] * (max_proteins - len(template_chain_ids))
    
    # Auto-generate protein_ids if not provided (AF3 requires uppercase letter chain IDs only)
    available_chain_ids = [c for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ" if c != binder_chain]
    for i in range(max_proteins):
        if not protein_ids[i] and (protein_seqs[i] or protein_msas[i] or template_chain_ids[i]):
            protein_ids[i] = available_chain_ids[i] if i < len(available_chain_ids) else available_chain_ids[i % len(available_chain_ids)]
    
    max_ligands = max(len(ligand_ids), len(ligand_smiles_list), len(ligand_ccds))
    ligand_ids = ligand_ids + [""] * (max_ligands - len(ligand_ids))
    ligand_smiles_list = ligand_smiles_list + [""] * (max_ligands - len(ligand_smiles_list))
    ligand_ccds = ligand_ccds + [""] * (max_ligands - len(ligand_ccds))
    
    max_nucleic = max(len(nucleic_types), len(nucleic_ids), len(nucleic_seqs))
    nucleic_types = nucleic_types + ["dna"] * (max_nucleic - len(nucleic_types))
    nucleic_ids = nucleic_ids + [""] * (max_nucleic - len(nucleic_ids))
    nucleic_seqs = nucleic_seqs + [""] * (max_nucleic - len(nucleic_seqs))
    
    mode = "binder"
    print("protein_msa", protein_msas)
    print("template_paths", template_paths)
    print("template_chain_ids", template_chain_ids)
    
    # Handle template: if single template path provided with multiple chain IDs, replicate the path
    unique_template_paths = [p for p in template_paths if p]
    if len(unique_template_paths) == 1 and max_proteins > 1:
        template_paths = [unique_template_paths[0]] * max_proteins
    
    # Convert PDB templates to CIF and extract sequences if needed
    template_sequences = {}
    converted_template_paths = list(template_paths)
    single_chain_template_paths = [""] * max_proteins  # Store paths to single-chain CIFs
    
    # Create temp folder for single-chain template CIFs
    if output_dir:
        template_temp_dir = os.path.join(output_dir, "template_cifs")
        os.makedirs(template_temp_dir, exist_ok=True)
    else:
        template_temp_dir = None
    
    for i, tpath in enumerate(template_paths):
        if tpath:
            # Convert PDB to CIF if needed
            if tpath.lower().endswith('.pdb'):
                cif_path = tpath.rsplit('.', 1)[0] + '.cif'
                if not os.path.exists(cif_path):
                    print(f"Converting template PDB to CIF: {tpath} -> {cif_path}")
                    pdb_to_cif(tpath, cif_path)
                converted_template_paths[i] = cif_path
            
            # Extract sequences from the template structure
            if converted_template_paths[i] not in template_sequences:
                template_sequences[converted_template_paths[i]] = sequence_from_structure(converted_template_paths[i])
            
            # Extract single chain for AF3 template (AF3 requires single-chain templates)
            tchain = template_chain_ids[i] if i < len(template_chain_ids) else ""
            if tchain:
                # Save single-chain CIF in output_dir/template_cifs if specified
                if template_temp_dir:
                    template_basename = os.path.basename(converted_template_paths[i]).rsplit('.', 1)[0]
                    single_chain_cif = os.path.join(template_temp_dir, f'{template_basename}_chain_{tchain}.cif')
                else:
                    single_chain_cif = converted_template_paths[i].rsplit('.', 1)[0] + f'_chain_{tchain}.cif'
                
                if not os.path.exists(single_chain_cif):
                    print(f"Extracting chain {tchain} from template: {converted_template_paths[i]} -> {single_chain_cif}")
                    extract_chain_to_cif(converted_template_paths[i], tchain, single_chain_cif)
                else:
                    # Ensure existing file has release date
                    add_release_date_to_cif(single_chain_cif)
                single_chain_template_paths[i] = single_chain_cif
    
    # Extract sequences from templates if protein_seqs not provided
    for i in range(max_proteins):
        if not protein_seqs[i] and not protein_msas[i]:
            tpath = converted_template_paths[i] if i < len(converted_template_paths) else ""
            tchain = template_chain_ids[i] if i < len(template_chain_ids) else ""
            if tpath and tchain and tpath in template_sequences:
                if tchain in template_sequences[tpath]:
                    protein_seqs[i] = template_sequences[tpath][tchain]
                    print(f"Extracted sequence for chain {tchain} from template: {protein_seqs[i][:50]}...")
    
    # Update template_paths to use converted CIF paths
    template_paths = converted_template_paths
    
    # Determine mode based on whether we have any targets (now including template-derived sequences)
    has_targets = any(protein_seqs) or any(protein_msas) or any(ligand_smiles_list) or any(ligand_ccds) or any(nucleic_seqs)
    if not has_targets:
        mode = "unconditional"
    
    # Extract sequences from MSAs if needed
    for i in range(len(protein_seqs)):
        if protein_msas[i] and not protein_seqs[i]:
            protein_seqs[i] = target_sequence_from_msa(protein_msas[i])

    sequences = []
    # Generate binder sequence with X residues if not provided

    if binder_sequence is None or binder_sequence == "":
        if binder_type == "protein":
            aas = "ACDEFGHIKLMNQRSTVWY"
            num_x = round(design_protein_length * percent_X / 100)
            pool = aas if aas else "X"
            seq_list = ["X"] * num_x + random.choices(pool, k=design_protein_length - num_x)
            random.shuffle(seq_list)
            binder_sequence = "".join(seq_list)
        elif binder_type == "dna":
            nucleotides = "ACGT"
            num_x = round(design_protein_length * percent_X / 100)
            seq_list = ["N"] * num_x + random.choices(nucleotides, k=design_protein_length - num_x)
            random.shuffle(seq_list)
            binder_sequence = "".join(seq_list)
        elif binder_type == "rna":
            nucleotides = "ACGU"
            num_x = round(design_protein_length * percent_X / 100)
            seq_list = ["N"] * num_x + random.choices(nucleotides, k=design_protein_length - num_x)
            random.shuffle(seq_list)
            binder_sequence = "".join(seq_list)
        else:
            raise ValueError(f"Invalid binder_type: {binder_type}. Must be 'protein', 'dna', or 'rna'")
    # Add binder (always present)
    if binder_type == "protein":
        sequences.append({
            "protein": {
                "id": binder_chain,
                "sequence": binder_sequence,
                "unpairedMsa": binder_msa if binder_msa else "",
                "pairedMsa": "",
                "templates": [],
                "unpairedMsaPath": "",
            }
        })
    elif binder_type == "dna":
        sequences.append({
            "dna": {
                "id": binder_chain.split(",")[0],
                "sequence": binder_sequence.split("/")[0] if "/" in binder_sequence else binder_sequence,
            }
        })
        sequences.append({
            "dna": {
                "id": binder_chain.split(",")[1],
                "sequence": binder_sequence.split("/")[1] if "/" in binder_sequence else binder_sequence,
            }
        })
    elif binder_type == "rna":
        sequences.append({
            "rna": {
                "id": binder_chain,
                "sequence": binder_sequence,
                "unpairedMsa": f">query\n{binder_sequence}\n",
                "unpairedMsaPath": "",
            }
        })
        
    else:
        raise ValueError(f"Invalid binder_type: {binder_type}. Must be 'protein', 'dna', or 'rna'")
    
    if mode == "binder":
        # Add all target proteins
        for i in range(max_proteins):
            if protein_seqs[i] or protein_msas[i]:
                # protein_ids should already be auto-generated, but fallback to valid letter just in case
                chain_id = protein_ids[i] if protein_ids[i] else available_chain_ids[i % len(available_chain_ids)]
                protein_entry = {
                    "protein": {
                        "id": chain_id,
                        "sequence": protein_seqs[i],
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                        "unpairedMsaPath": protein_msas[i] if protein_msas[i] else "",
                    }
                }
                
                # Add template if specified (use single-chain CIF path)
                single_chain_cif = single_chain_template_paths[i] if i < len(single_chain_template_paths) else ""
                tchain = template_chain_ids[i] if i < len(template_chain_ids) else ""
                if single_chain_cif and os.path.exists(single_chain_cif):
                    single_chain_cif_abs = os.path.abspath(single_chain_cif)
                    seq_len = len(protein_seqs[i])
                    protein_entry["protein"]["templates"] = [{
                        "mmcifPath": single_chain_cif_abs,
                        "queryIndices": list(range(seq_len)),
                        "templateIndices": list(range(seq_len)),
                    }]
                    print(f"Added template for chain {protein_ids[i]} from {single_chain_cif_abs} (seq_len={seq_len})")
                
                sequences.append(protein_entry)
        
        # Add all ligands (use letters starting after proteins for auto-generated IDs)
        ligand_chain_offset = max_proteins
        for i in range(max_ligands):
            ligand_chain_id = ligand_ids[i] if ligand_ids[i] else available_chain_ids[(ligand_chain_offset + i) % len(available_chain_ids)]
            if ligand_smiles_list[i]:
                sequences.append({
                    "ligand": {
                        "id": ligand_chain_id,
                        "smiles": ligand_smiles_list[i],
                    }
                })
            elif ligand_ccds[i]:
                sequences.append({
                    "ligand": {
                        "id": ligand_chain_id,
                        "ccdCodes": [ligand_ccds[i]],
                    }
                })
        
        # Add all nucleic acids (use letters starting after proteins and ligands for auto-generated IDs)
        nucleic_chain_offset = max_proteins + max_ligands
        for i in range(max_nucleic):
            if nucleic_seqs[i]:
                nucleic_chain_id = nucleic_ids[i] if nucleic_ids[i] else available_chain_ids[(nucleic_chain_offset + i) % len(available_chain_ids)]
                sequences.append({
                    nucleic_types[i]: {
                        "id": nucleic_chain_id,
                        "sequence": nucleic_seqs[i],
                    }
                })
    
    # Sort sequences by chain ID for consistency
    sequences = sorted(
        sequences, 
        key=lambda entry: list(entry.values())[0]["id"]
    )

    # Build the data dictionary
    data = [{
        "name": f"design_{mode}",
        "sequences": sequences,
        "modelSeeds": [model_seed if model_seed is not None else np.random.randint(low=1, high=1e6)],
        "dialect": "alphafold3",
        "version": 1,
    }]
    
    return json.dumps(data)