import os
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


def convert_cif_files_from_prediction_folder(results_dir, save_dir):
    """
    Convert all .cif files in results directory to .pdb format and save them in save directory.
    
    Args:
        results_dir (str): Path to directory containing .cif files in nested subdirectories
        save_dir (str): Path to directory where converted .pdb files will be saved
    
    Returns:
        int: Number of files converted
    """
    count = 0
    os.makedirs(save_dir, exist_ok=True)
    
    for subfolder in os.listdir(results_dir):
        subfolder_path = os.path.join(results_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for subdir in os.listdir(subfolder_path):
                subdir_path = os.path.join(subfolder_path, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if os.path.isdir(os.path.join(subdir_path, file)):
                            for file2 in os.listdir(os.path.join(subdir_path, file)):
                                if file2.endswith('.cif'):
                                    count += 1
                                    cif_path = os.path.join(subdir_path, file, file2)
                                    pdb_path = os.path.join(save_dir, file2.replace('.cif', '.pdb'))
                                    convert_cif_to_pdb(cif_path, pdb_path)
                                    print(pdb_path)
    return count

def convert_cif_files(results_dir, save_dir):
    """
    Convert all .cif files in results directory to .pdb format and save them in save directory.
    
    Args:
        results_dir (str): Path to directory containing .cif files
        save_dir (str): Path to directory where converted .pdb files will be saved
    
    Returns:
        int: Number of files converted
    """
    count = 0
    os.makedirs(save_dir, exist_ok=True)
    
    for file in os.listdir(results_dir):
        if file.endswith('.cif'):
            count += 1
            cif_path = os.path.join(results_dir, file)
            pdb_path = os.path.join(save_dir, file.replace('.cif', '.pdb'))
            convert_cif_to_pdb(cif_path, pdb_path)
            print(pdb_path)
    return count