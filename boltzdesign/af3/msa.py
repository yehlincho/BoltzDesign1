import logging
import os
import random
import tarfile
import time
import json
import base64
import subprocess
from typing import Optional, Union, Dict
from pathlib import Path
from urllib import request, parse, error
from tqdm import tqdm
from boltz_ph.constants import RNA_CHAIN_POLY_TYPE

logger = logging.getLogger(__name__)


TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

import os
import shutil
import json
import logging
import sys
import numpy as np
import pandas as pd
import urllib
from tqdm import tqdm

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Selection


# AlphaFold3 Database Info (to be used by downstream modules)
RNA_DATABASE_INFO = {
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta": "RNAcentral",
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta": "NT_RNA",
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta": "Rfam",
}
AF3_SOURCE = "https://storage.googleapis.com/alphafold-databases/v3.0"

# --- Chain Conversion Helpers ---

class OutOfChainsError(Exception):
    pass

def int_to_chain(i: int) -> str:
    """Convert an integer to a one-letter chain ID (A-Z, a-z, 0-9)."""
    if i < 26:
        return chr(ord("A") + i)
    elif i < 52:
        return chr(ord("a") + i - 26)
    elif i < 62:
        return chr(ord("0") + i - 52)
    else:
        raise OutOfChainsError


def rename_chains(structure):
    """
    Renames chains to be one-letter valid PDB chains (A-Z, a-z, 0-9).
    Existing one-letter chains are kept. Others are renamed uniquely.
    Returns a map between new and old chain IDs.
    """
    next_chain = 0
    chainmap = {c.id: c.id for c in structure.get_chains() if len(c.id) == 1}

    # Helper function to find the next available one-letter chain
    def get_next_chain():
        nonlocal next_chain
        while True:
            try:
                c = int_to_chain(next_chain)
                if c not in chainmap:
                    return c
            except OutOfChainsError:
                raise
            next_chain += 1

    for o in structure.get_chains():
        if len(o.id) != 1:
            try:
                c = get_next_chain()
                chainmap[c] = o.id
                o.id = c
            except OutOfChainsError as e:
                logging.error("Too many chains to represent in PDB format")
                raise e
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
                    logging.warning(
                        f"Truncating residue name '{resname}' to '{truncated}'"
                    )
                    residue.resname = truncated

# --- Core Conversion Logic ---

def convert_cif_to_pdb(ciffile, pdbfile):
    """
    Convert a CIF file to PDB format, handling chain renaming and residue name truncation.
    """
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)

    # Use a dummy structure ID if path is not an ID
    strucid = os.path.basename(ciffile).split('.')[0] if len(os.path.basename(ciffile)) > 4 else "1xxx"

    # Parse CIF file
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(strucid, ciffile)
    except Exception as e:
        logging.error(f"Failed to parse CIF file {ciffile}: {e}")
        return False

    # Rename chains
    try:
        rename_chains(structure)
    except OutOfChainsError:
        return False

    # Truncate long ligand or residue names
    sanitize_residue_names(structure)

    # Write to PDB
    io = PDBIO()
    io.set_structure(structure)
    try:
        io.save(pdbfile)
    except Exception as e:
        logging.error(f"Failed to write PDB file {pdbfile}: {e}")
        return False
        
    return True


def convert_cif_files_to_pdb(
    results_dir: str, save_dir: str, af_dir: bool = False, high_iptm: bool = False, i_ptm_cutoff: float = 0.5
):
    """
    Convert all .cif files in results_dir to .pdb format and save in save_dir.
    Filters by i-pTM score if high_iptm is True.
    """
    confidence_scores = []
    os.makedirs(save_dir, exist_ok=True)
    
    # Find all result files that match the pattern
    cif_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            is_af_file = af_dir and file.endswith("_model.cif")
            is_boltz_file = (not af_dir) and file.endswith(".cif")
            if is_af_file or is_boltz_file:
                cif_files.append((root, file))

    if not cif_files:
        print(f"No CIF files found in {results_dir} matching pattern (af_dir={af_dir}).")
        return

    for root, file in cif_files:
        cif_path = os.path.join(root, file)
        pdb_path = os.path.join(save_dir, file.replace(".cif", ".pdb"))

        # Skip if conversion already done
        if os.path.exists(pdb_path):
             # Try to check if confidence data exists for existing files if in high_iptm mode
            if high_iptm:
                score_name = file.replace(".cif", ".cif")
                if any(score.get('file') == score_name for score in confidence_scores):
                     continue # Already processed and score recorded
            else:
                continue

        iptm = float("-inf")
        plddt = float("-inf")
        should_convert = True

        if high_iptm:
            try:
                # AlphaFold3 output format
                if af_dir:
                    base_name = file.replace("_model.cif", "")
                    confidence_file_summary = os.path.join(root, f"{base_name}_summary_confidences.json")
                    confidence_file = os.path.join(root, f"{base_name}_confidences.json")
                # Boltz output format
                else:
                    confidence_file_summary = os.path.join(root, "confidence_summary.json") # Example name
                    confidence_file = os.path.join(root, "confidence_full.json") # Example name

                
                # Try loading summary confidence data for ipTM
                if os.path.exists(confidence_file_summary):
                    with open(confidence_file_summary) as f:
                        confidence_data = json.load(f)
                        iptm = confidence_data.get("iptm", float("-inf"))
                
                # Try loading full confidence data for pLDDT
                if os.path.exists(confidence_file):
                    with open(confidence_file) as f:
                        confidence_data = json.load(f)
                        plddt = np.mean(confidence_data.get("atom_plddts", [0.0]))

                if iptm < i_ptm_cutoff:
                    should_convert = False
                    print(f"Skipping {file}: i-pTM ({iptm:.2f}) below threshold ({i_ptm_cutoff:.2f}).")

            except Exception as e:
                print(f"WARNING: Could not read confidence data for {file}: {e}")
                should_convert = True # Fail open

        if should_convert:
            print(f"Converting {cif_path} (i-pTM: {iptm:.2f})...")
            if convert_cif_to_pdb(cif_path, pdb_path):
                if high_iptm:
                    confidence_scores.append({"file": file, "iptm": iptm, "plddt": plddt})
            else:
                print(f"❌ Failed to convert {cif_path}.")


    if confidence_scores:
        confidence_scores_path = os.path.join(save_dir, "high_iptm_confidence_scores.csv")
        pd.DataFrame(confidence_scores).to_csv(confidence_scores_path, index=False)
        print(f"✅ Saved confidence scores to {confidence_scores_path}")


import os
import urllib.request
from tqdm import tqdm

def download_with_progress(url, dest_path):
    """Download a file with a progress bar (no requests dependency)"""
    try:
        with urllib.request.urlopen(url) as response:
            file_size = int(response.headers.get("Content-Length", 0))
            desc = f"Downloading {os.path.basename(dest_path)}"

            with tqdm(total=file_size, unit="B", unit_scale=True, desc=desc) as pbar:
                with open(dest_path, "wb") as out_file:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        pbar.update(len(chunk))
        return True

    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False



def run_mmseqs2(
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    auth_headers: Optional[Dict[str, str]] = None,
) -> tuple[list[str], list[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_header_auth = auth_headers is not None
    if has_basic_auth and (has_header_auth or auth_headers):
        raise ValueError(
            "Cannot use both basic authentication (username/password) and header/API key authentication."
        )

    # Set up headers
    # NOTE: the public colabfold MSA server (api.colabfold.com) tarpits/blocks the
    # "boltz" User-Agent (requests hang until timeout). Any other UA works normally.
    headers = {"User-Agent": "ProteinHunter/1.0"}

    # Set up authentication
    if has_basic_auth:
        auth_str = f"{msa_server_username}:{msa_server_password}"
        encoded_auth = base64.b64encode(auth_str.encode()).decode()
        headers["Authorization"] = f"Basic {encoded_auth}"
        logger.debug(f"MMSeqs2 server authentication: using basic auth for user '{msa_server_username}'")
    elif has_header_auth:
        headers.update(auth_headers)
        logger.debug("MMSeqs2 server authentication: using header-based authentication")
    else:
        logger.debug("MMSeqs2 server authentication: no credentials provided")

    def _http_request(url, data=None, method='GET'):
        """Standard library replacement for requests.get/post"""
        req_data = parse.urlencode(data).encode() if data else None
        req = request.Request(url, data=req_data, headers=headers, method=method)
        with request.urlopen(req, timeout=60) as response:
            return response.read(), response.getcode()

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        error_count = 0
        while True:
            try:
                logger.debug(f"Submitting MSA request to {host_url}/{submission_endpoint}")
                body, status_code = _http_request(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    method='POST'
                )
                return json.loads(body.decode())
            except Exception as e:
                error_count += 1
                is_429 = "429" in str(e)
                wait = min(60 * 2 ** (error_count - 1), 600) if is_429 else 5
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/10)")
                logger.warning(f"Error: {e}")
                if error_count > 10:
                    raise Exception("Too many failed attempts for the MSA generation request.")
                logger.warning(f"Waiting {wait}s before retry...")
                time.sleep(wait)

    def status(ID):
        error_count = 0
        while True:
            try:
                logger.debug(f"Checking MSA job status for ID: {ID}")
                body, _ = _http_request(f"{host_url}/ticket/{ID}")
                return json.loads(body.decode())
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/10)")
                if error_count > 10:
                    raise Exception("Too many failed attempts for the MSA generation request.")
                time.sleep(5)

    def download(ID, path):
        error_count = 0
        while True:
            try:
                logger.debug(f"Downloading MSA results for ID: {ID}")
                body, _ = _http_request(f"{host_url}/result/download/{ID}")
                with open(path, "wb") as out:
                    out.write(body)
                break
            except Exception as e:
                error_count += 1
                logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/10)")
                if error_count > 10:
                    raise Exception("Too many failed attempts for the MSA generation request.")
                time.sleep(5)

    # Process input x
    seqs = [x] if isinstance(x, str) else x

    # Setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # Define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # Call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # Deduplicate and keep track of order
    seqs_unique = []
    [seqs_unique.append(seq) for seq in seqs if seq not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")
                out = submit(seqs_unique, mode, N)
                
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] in ["ERROR", "MAINTENANCE"]:
                    raise Exception(f"MMseqs2 API error: {out['status']}. Please try again later.")

                ID, TIME = out["id"], 0
                logger.debug(f"MSA job submitted successfully with ID: {ID}")
                pbar.set_description(out["status"])
                
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False
                elif out["status"] == "ERROR":
                    raise Exception("MMseqs2 API returned an error during processing.")

            download(ID, tar_gz_file)

    # Prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # Extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # Gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        with open(a3m_file, "r") as f:
            for line in f:
                if len(line) > 0:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines:
                            a3m_lines[M] = []
                    a3m_lines[M].append(line)

    return ["".join(a3m_lines[n]) for n in Ms]


def process_msa(chain_id: str, sequence: str, msa_dir: Path) -> Path:
    """Process MSA for a single chain using MMseqs2 and return path to .a3m file."""
    msa_chain_dir = msa_dir / f"{chain_id}"
    env_dir = msa_chain_dir.with_name(f"{msa_chain_dir.name}_env")
    env_dir.mkdir(exist_ok=True, parents=True)

    unpaired_msa = run_mmseqs2(
        [sequence],
        str(msa_chain_dir),
        use_env=True,
        use_pairing=False,
        host_url="https://api.colabfold.com",
        pairing_strategy="greedy",
    )

    msa_a3m_path = env_dir / "msa.a3m"
    msa_a3m_path.write_text(unpaired_msa[0])

    return msa_a3m_path



def download_selected_databases(database_settings: dict, afdb_dir: str):
    """Download only the databases selected in the settings."""
    afdb_dir = os.path.expanduser(afdb_dir)
    os.makedirs(afdb_dir, exist_ok=True)
    
    selected_db_files = [
        db_file for db_file, db_key in RNA_DATABASE_INFO.items()
        if database_settings.get(db_key, False)
    ]

    if not selected_db_files:
        print("⚠️ No RNA databases selected for download!")
        return

    print(
        f"🌐 Downloading {len(selected_db_files)} RNA databases: {', '.join([RNA_DATABASE_INFO[db] for db in selected_db_files])}"
    )

    missing_dbs = []
    for db in selected_db_files:
        db_path = os.path.join(afdb_dir, db)
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            missing_dbs.append(db)

    if not missing_dbs:
        print("✅ All selected databases already downloaded.")
        return

    with tqdm(
        total=len(missing_dbs), desc="Overall progress", unit="db", position=0
    ) as main_pbar:
        for db in missing_dbs:
            dest_path = os.path.join(afdb_dir, f"{db}.zst")
            final_path = os.path.join(afdb_dir, db)

            print(f"📥 Downloading {db} ({RNA_DATABASE_INFO[db]})...")
            url = f"{AF3_SOURCE}/{db}.zst"
            if download_with_progress(url, dest_path):
                print(f"📦 Decompressing {db}...")
                try:
                    # Decompress with zstd (assumes zstd is in PATH)
                    subprocess.run(
                        ["zstd", "--decompress", "-f", dest_path, "-o", final_path],
                        check=True,
                    )
                    print(f"✅ Successfully processed {db}")
                    os.remove(dest_path)
                except Exception as e:
                    print(f"❌ Error decompressing {db}: {e}")

            main_pbar.update(1)




def af3_generate_rna_msa(rna_sequence: str, database_settings: dict, afdb_dir: str, hmmer_path: str) -> str:
    print(f"Generating MSA for {rna_sequence}...")
    """Generate MSA for an RNA sequence using the AlphaFold3 HMMER pipeline."""
    
    rna_sequence = rna_sequence.upper().strip().replace('T', 'U') # RNA must be U, not T
    valid_bases = set("ACGU")
    if not all(base in valid_bases for base in rna_sequence):
        raise ValueError(
            f"Invalid RNA sequence. Must contain only A, C, G, U: {rna_sequence}"
        )

    # Setup paths to binaries and databases
    hmmer_path = os.path.expanduser(hmmer_path)
    afdb_dir = os.path.expanduser(afdb_dir)
    print("hmmer_path: ", hmmer_path)
    print("afdb_dir: ", afdb_dir)

    nhmmer_binary = os.path.join(hmmer_path, "bin/nhmmer")
    hmmalign_binary = os.path.join(hmmer_path, "bin/hmmalign")
    hmmbuild_binary = os.path.join(hmmer_path, "bin/hmmbuild")
    print("nhmmer_binary: ", nhmmer_binary)
    print("hmmalign_binary: ", hmmalign_binary)
    print("hmmbuild_binary: ", hmmbuild_binary)

    

    database_paths = {
        "Rfam": os.path.join(afdb_dir, RNA_DATABASE_INFO["rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"]),
        "RNAcentral": os.path.join(afdb_dir, RNA_DATABASE_INFO["rnacentral_active_seq_id_90_cov_80_linclust.fasta"]),
        "NT_RNA": os.path.join(afdb_dir, RNA_DATABASE_INFO["nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"]),
    }
    

    # 1. Download missing databases
    selected_db_keys = [db_key for db_file, db_key in RNA_DATABASE_INFO.items() if database_settings.get(db_key, False)]
    download_selected_databases(database_settings, afdb_dir)
    
    # 2. Filter databases to only existing, selected ones
    filtered_db_paths = {}
    for db_key in selected_db_keys:
        db_path = database_paths.get(db_key)
        if db_path and os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            filtered_db_paths[db_key] = db_path

    if not filtered_db_paths:
        print("❌ No selected RNA databases found or none selected. Returning query only.")
        return f">query\n{rna_sequence}\n"

    print(f"🔍 Will search {len(filtered_db_paths)} databases.")
    
    # 3. Run Nhmmer on each database
    msas = []
    rna_msa_start_time = time.time()
    
    with tqdm(total=len(filtered_db_paths), desc="Database searches", unit="db") as progress_bar:
        for db_name, db_path in filtered_db_paths.items():
            nhmmer_runner = Nhmmer(
                binary_path=nhmmer_binary,
                hmmalign_binary_path=hmmalign_binary,
                hmmbuild_binary_path=hmmbuild_binary,
                database_path=db_path,
                n_cpu=database_settings.get("n_cpu", 2),
                e_value=database_settings.get("e_value", 0.001),
                max_sequences=database_settings.get("max_sequences_per_db", 10000),
                alphabet="rna",
                time_limit_minutes=database_settings.get("time_limit_minutes"),
            )
            try:
                a3m_result = nhmmer_runner.query(rna_sequence)
                msa = Msa.from_a3m(
                    query_sequence=rna_sequence,
                    chain_poly_type=RNA_CHAIN_POLY_TYPE,
                    a3m=a3m_result,
                    deduplicate=False,
                )
                msas.append(msa)
                print(f"✅ Found {msa.depth} sequences in {db_name}")
            except Exception as e:
                print(f"❌ Error processing {db_name}: {e}")
            progress_bar.update(1)

    # 4. Merge and deduplicate MSAs
    if not msas:
        print("⚠️ No homologous sequences found. MSA contains only the query sequence.")
        a3m = f">query\n{rna_sequence}\n"
    else:
        rna_msa = Msa.from_multiple_msas(msas=msas, deduplicate=True)
        print(f"🎉 MSA construction complete! Found {rna_msa.depth} unique sequences.")
        a3m = rna_msa.to_a3m()

    elapsed_time = time.time() - rna_msa_start_time
    print(f"⏱️ Total RNA MSA generation time: {elapsed_time:.2f} seconds")

    return a3m
