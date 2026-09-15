"""
AlphaFold3 Validator for cross-validation.

This module provides AF3-based validation for protein designs
from any source (Boltz, Chai, or other).

Uses the AF3 conda environment directly (no Docker required).

IMPORTANT: alphafold3.cpp must be imported at module level before other imports.
This uses the conda-installed alphafold3 package (not ~/alphafold3/src).
"""

import os
import sys

# Import alphafold3.cpp EARLY from conda environment (required for proper initialization)
# NOTE: Do NOT add ~/alphafold3/src to path - use the conda-installed package
try:
    import alphafold3.cpp
    AF3_AVAILABLE = True
except ImportError as e:
    AF3_AVAILABLE = False
    AF3_IMPORT_ERROR = str(e)

import gc
import json
import pathlib
import datetime
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from .base import BaseValidator, ValidationInput, ValidationResult

# Default paths for AF3 (can be overridden)
_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
# Match af3_ph default: $HOME/models
_DEFAULT_MODEL_DIR = _HOME_DIR / 'alphafold3' / 'models'
_DEFAULT_DB_DIR = _HOME_DIR / 'alphafold3' / 'alphafold3_data_save'


def ensure_cif_has_data_field(cif_path: Path) -> Path:
    """
    Ensure a CIF file starts with the data_ field.
    If it doesn't, add it based on the filename.
    
    Similar to how af3_ph handles template CIF files.
    
    Args:
        cif_path: Path to CIF file
        
    Returns:
        Path to the (possibly fixed) CIF file
    """
    cif_path = Path(cif_path)
    
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    # Read the file
    with open(cif_path, 'r') as f:
        content = f.read()
    
    # Check if file is empty
    if not content.strip():
        raise ValueError(f"CIF file is empty: {cif_path}")
    
    # Check if it starts with data_ (case-insensitive, allowing whitespace)
    content_stripped = content.strip()
    if content_stripped.lower().startswith('data_'):
        return cif_path  # Already has data_ field
    
    # If not, add the data_ field
    # Extract data name from filename (remove extension)
    data_name = cif_path.stem
    # Remove any invalid characters for CIF data names
    # CIF data names should be alphanumeric and underscore, max 4 chars typically
    data_name = ''.join(c for c in data_name if c.isalnum() or c == '_')
    # Limit length and ensure it starts with a letter or number
    if not data_name:
        data_name = 'DATA'
    elif not data_name[0].isalnum():
        data_name = 'DATA_' + data_name
    # Limit to reasonable length (CIF data names are typically short)
    if len(data_name) > 20:
        data_name = data_name[:20]
    
    # Add data_ field at the beginning
    # Preserve any leading whitespace/newlines, but ensure data_ is first non-whitespace
    lines = content.split('\n')
    # Find first non-empty line
    first_non_empty_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            first_non_empty_idx = i
            break
    
    # Insert data_ line before first non-empty line
    new_lines = lines[:first_non_empty_idx] + [f"data_{data_name}", "#"] + lines[first_non_empty_idx:]
    new_content = '\n'.join(new_lines)
    
    # Write back to file
    with open(cif_path, 'w') as f:
        f.write(new_content)
    
    print(f"  ✓ Fixed CIF file: added data_{data_name} field")
    return cif_path


def prepare_template_for_af3(template_path: str, output_dir: Optional[Path] = None) -> Path:
    """
    Prepare a template file for AF3 validation.
    Converts PDB to CIF if needed and ensures proper CIF format.
    
    Similar to af3_ph's prepare_template function.
    
    Args:
        template_path: Path to template PDB or CIF file
        output_dir: Directory for converted/fixed CIF file (optional)
        
    Returns:
        Path to the prepared CIF file
    """
    template_path_str = str(template_path).strip()

    # Resolve 4-letter PDB codes by downloading from RCSB
    if len(template_path_str) == 4 and template_path_str.isalnum():
        local_cif = Path(f"{template_path_str}.cif")
        if not local_cif.exists():
            import urllib.request
            url = f"https://files.rcsb.org/download/{template_path_str}.cif"
            print(f"  📥 Downloading template {template_path_str} from RCSB...")
            urllib.request.urlretrieve(url, local_cif)
        template_path = local_cif
    else:
        template_path = Path(template_path_str)

    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Convert PDB to CIF if needed
    if template_path.suffix.lower() == '.pdb':
        try:
            # Import BioPython components
            from Bio.PDB import PDBParser, MMCIFIO
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                cif_path = output_dir / f"{template_path.stem}.cif"
            else:
                cif_path = template_path.with_suffix('.cif')
            
            print(f"  Converting PDB to CIF: {template_path.name} -> {cif_path.name}")
            
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("structure", str(template_path))
            
            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(cif_path))
            
            # Ensure it has data_ field
            ensure_cif_has_data_field(cif_path)
            
            # Add required release date (similar to af3_ph)
            try:
                from .af_utils import add_release_date_to_cif
                add_release_date_to_cif(str(cif_path))
            except Exception as e:
                print(f"  Warning: Could not add release date to CIF: {e}")
            
            return cif_path
        except ImportError:
            raise ImportError("BioPython is required for PDB to CIF conversion. Install with: pip install biopython")
        except Exception as e:
            raise ValueError(f"Failed to convert PDB to CIF: {e}")
    else:
        # It's already a CIF file, just ensure it has data_ field
        cif_path = template_path
        ensure_cif_has_data_field(cif_path)
        
        # Add required release date if not present
        try:
            from .af_utils import add_release_date_to_cif
            add_release_date_to_cif(str(cif_path))
        except Exception as e:
            print(f"  Warning: Could not add release date to CIF: {e}")
        
        return cif_path


def ipSAE(
    *,
    num_tokens: int,
    asym_ids: np.ndarray,
    full_pae: np.ndarray,
    mask: np.ndarray = None,
    pae_cutoff: float = 10,
):
    """Calculate interface predicted Structural Alignment Error (ipSAE)."""
    if mask is None:
        mask = np.ones(shape=full_pae.shape[1:], dtype=bool)
    
    full_pae = full_pae[:, :num_tokens, :num_tokens]
    mask = mask[:num_tokens, :num_tokens]
    asym_ids = asym_ids[:num_tokens]
    
    unique_asym_ids = np.unique(asym_ids)
    num_chains = len(unique_asym_ids)
    num_samples = full_pae.shape[0]
    ipsae = np.zeros((num_samples, num_chains, num_chains))
    
    def d0_from_L(L: np.ndarray) -> float:
        d0 = 1.24 * (np.cbrt(L - 15)) - 1.8
        return d0.clip(min=1)

    for idx1, asym_id_1 in enumerate(unique_asym_ids):
        subset = full_pae[:, asym_ids == asym_id_1, :]       
        subset_mask = mask[asym_ids == asym_id_1, :]         
        
        for idx2, asym_id_2 in enumerate(unique_asym_ids):
            subsubset = subset[:, :, asym_ids == asym_id_2]            
            subsubset_mask = subset_mask[:, asym_ids == asym_id_2]     
            
            subsubset_pae_mask = subsubset < pae_cutoff 
            A = np.sum(subsubset_pae_mask, axis=-1) > 0 
            B = np.sum(subsubset_pae_mask, axis=-2) > 0 
            L = A.sum(axis=1) + B.sum(axis=1)
            d0 = d0_from_L(L)
            
            tmp = 1.0 / (1.0 + (subsubset / d0[:, None, None]) ** 2)  
            numer = (tmp * subsubset_pae_mask).sum(axis=-1) 
            counts = subsubset_pae_mask.sum(axis=-1)          
            
            res_val = np.divide(numer, counts, out=np.zeros_like(numer, dtype=float), where=counts > 0) 
            ipsae[:, idx1, idx2] = np.max(res_val, axis=-1)

    ipsae_min = np.minimum(ipsae, np.transpose(ipsae, (0, 2, 1)))
    ipsae_max = np.maximum(ipsae, np.transpose(ipsae, (0, 2, 1)))
    ipsae_mean = 0.5 * (ipsae + np.transpose(ipsae, (0, 2, 1)))

    return ipsae_min, ipsae_max, ipsae_mean



class AF3Validator(BaseValidator):
    """
    AlphaFold3-based validator for cross-validation.
    
    This validator runs AF3 directly using the conda environment
    (no Docker required).
    
    Usage:
        validator = AF3Validator(
            output_dir="./validation_results",
            model_dir="/path/to/alphafold3/models",
        )
        
        input_data = ValidationInput.from_csv(
            "designs.csv", 
            target_seq="MVKL...",
        )
        
        results = validator.validate_batch(input_data)
    """
    
    def __init__(self,
                 output_dir: str,
                 device: str = "0",  # GPU device ID
                 model_dir: str = None,
                 db_dir: str = None,
                 template_path: str = None,  # Explicit template path (like design module)
                 template_chain_id: str = None,  # Chain ID in CIF to use as target sequence source
                 msa_mode: str = "single",  # "mmseqs", "single", or path to .a3m file
                 msa_binder: bool = False,  # also generate an MSA for the binder (chain A) — matches AF3 default for antibodies
                 binder_template_path: str = None,  # antibody framework template for chain A (RFdiffusion-style antibody validation)
                 num_recycles: int = 10,
                 num_diffusion_samples: int = 1,
                 flash_attention_implementation: str = "xla",
                 seed: int = 1,
                 buckets: List[str] = None,
                 include_rosetta_metrics: bool = False,  # Whether to run PyRosetta relaxation and scoring
                 **kwargs):
        super().__init__(output_dir=output_dir, device=device, **kwargs)
        self.include_rosetta_metrics = include_rosetta_metrics

        # Set default paths (ensure they are Path objects)
        if model_dir:
            self.model_dir = pathlib.Path(model_dir)
        else:
            self.model_dir = _DEFAULT_MODEL_DIR if isinstance(_DEFAULT_MODEL_DIR, pathlib.Path) else pathlib.Path(_DEFAULT_MODEL_DIR)
        self.db_dir = pathlib.Path(db_dir) if db_dir else _DEFAULT_DB_DIR

        # Template and MSA mode settings (explicit like design module)
        self.template_path = template_path
        self.template_chain_id = template_chain_id  # which chain of the CIF is the target
        self.msa_mode = msa_mode  # "mmseqs", "single", or path to .a3m file
        self.msa_binder = msa_binder  # generate MSA for binder chain A too
        self.binder_template_path = binder_template_path  # antibody framework template for chain A
        
        # Track validation mode for CSV output
        self.validation_mode = self._determine_validation_mode()
        
        print(f"  Validation mode: {self.validation_mode}")
        if self.template_path:
            print(f"  Template: {self.template_path}")
        if self.msa_mode and self.msa_mode != "single":
            print(f"  MSA mode: {self.msa_mode}")
        
        self.num_recycles = num_recycles
        self.num_diffusion_samples = num_diffusion_samples
        self.flash_attention_implementation = flash_attention_implementation
        self.seed = seed
        self._skip_apo_below = None  # set by validate_batch from iptm_threshold
        self.buckets = buckets or ['256', '512', '768', '1024', '1280', '1536', 
                                    '2048', '2560', '3072', '3584', '4096', '4608', '5120']
        
        # Will be set on initialize
        self.predictor = None
        self.data_pipeline_config = None
        self.jax_device = None
    
    def _determine_validation_mode(self) -> str:
        """Determine validation mode based on template and MSA settings."""
        if self.template_path:
            return "template"
        elif self.msa_mode == "mmseqs":
            return "mmseqs"
        elif self.msa_mode and self.msa_mode != "single" and Path(self.msa_mode).exists():
            return "msa_file"
        else:
            return "single"
        
    @property
    def model_name(self) -> str:
        return "af3"
    
    def initialize(self) -> None:
        """Initialize AF3 model using conda environment."""
        # Check if AF3 is available
        if not AF3_AVAILABLE:
            raise ImportError(
                f"AlphaFold3 not available. Error: {AF3_IMPORT_ERROR}\n"
                "Make sure you're running in the af3 conda environment."
            )
        
        # Set GPU device
        gpu_id = self.device.replace("cuda:", "") if "cuda:" in self.device else self.device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Import AF3 modules (alphafold3.cpp already imported at module level)
        from alphafold3.data import pipeline
        from alphafold3.jax.attention import attention
        import jax
        
        # Import from af3_ph pipeline module
        af3_ph_dir = str(Path(__file__).parent.parent / "af3_ph")
        if af3_ph_dir not in sys.path:
            sys.path.insert(0, af3_ph_dir)
        from .runtime import get_symbols
        make_model_config, ModelRunner, _, _, _ = get_symbols(getattr(self, 'af3_root', None))
        
        # Setup JAX device
        gpu_devices = jax.devices('gpu')
        if gpu_devices:
            self.jax_device = gpu_devices[0]  # Use first available GPU after CUDA_VISIBLE_DEVICES
            print(f"✓ Using GPU: {self.jax_device}")
        else:
            self.jax_device = jax.devices('cpu')[0]
            print("⚠️  Warning: Using CPU (GPU not available)")
        
        # Setup data pipeline config
        max_template_date = datetime.date.fromisoformat('2025-01-01')
        
        self.data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=shutil.which('jackhmmer'),
            nhmmer_binary_path=shutil.which('nhmmer'),
            hmmalign_binary_path=shutil.which('hmmalign'),
            hmmsearch_binary_path=shutil.which('hmmsearch'),
            hmmbuild_binary_path=shutil.which('hmmbuild'),
            small_bfd_database_path=str(self.db_dir / 'bfd-first_non_consensus_sequences.fasta'),
            mgnify_database_path=str(self.db_dir / 'mgy_clusters_2022_05.fa'),
            uniprot_cluster_annot_database_path=str(self.db_dir / 'uniprot_all_2021_04.fa'),
            uniref90_database_path=str(self.db_dir / 'uniref90_2022_05.fa'),
            ntrna_database_path=str(self.db_dir / 'nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta'),
            rfam_database_path=str(self.db_dir / 'rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta'),
            rna_central_database_path=str(self.db_dir / 'rnacentral_active_seq_id_90_cov_80_linclust.fasta'),
            pdb_database_path=str(self.db_dir / 'mmcif_files'),
            seqres_database_path=str(self.db_dir / 'pdb_seqres_2022_09_28.fasta'),
            jackhmmer_n_cpu=1,
            nhmmer_n_cpu=1,
            max_template_date=max_template_date,
        )
        
        # Initialize model runner
        # Ensure model_dir is a Path object (not string)
        model_dir_path = pathlib.Path(self.model_dir) if not isinstance(self.model_dir, pathlib.Path) else self.model_dir
        
        print(f"✓ Initializing AF3 model from: {model_dir_path}")
        print(f"  Flash attention: {self.flash_attention_implementation}")
        print(f"  Num recycles: {self.num_recycles}")
        print(f"  Num diffusion samples: {self.num_diffusion_samples}")
        
        self.predictor = ModelRunner(
            config=make_model_config(
                flash_attention_implementation=self.flash_attention_implementation,
                num_diffusion_samples=self.num_diffusion_samples,
                num_recycles=self.num_recycles,
                return_embeddings=False,
            ),
            device=self.jax_device,
            model_dir=model_dir_path,
        )
        
        self._initialized = True
        print(f"✓ AF3 validator initialized on GPU {gpu_id}")
    
    def _extract_seq_from_template(self, template_cif_path: Path) -> Optional[str]:
        """Extract sequence from the specified chain of the template CIF."""
        try:
            import gemmi
            st = gemmi.read_structure(str(template_cif_path))
            chain_id = self.template_chain_id
            if chain_id:
                chain = st[0][chain_id]
            else:
                chain = next(iter(st[0]))  # first chain
            # get_polymer() fails after PDB→CIF conversion (entity type lost → Unknown).
            # Iterate all residues and filter out non-amino-acid entries via one_letter_code.
            seq = "".join(
                gemmi.find_tabulated_residue(r.name).one_letter_code for r in chain
            ).replace("?", "")
            print(f"  📄 Extracted target seq from template chain {chain.name} ({len(seq)} aa)")
            return seq
        except Exception as e:
            print(f"  ⚠️  Could not extract sequence from template: {e}")
            return None

    def _run_mmseqs(self, target_seq: str, chain_id: str = "B") -> Optional[str]:
        """Run MMseqs2 to generate MSA for target sequence.
        
        Uses a hash of the sequence to create unique cache directories,
        preventing conflicts when validating different multi-chain targets.
        """
        try:
            from .msa import process_msa
            import hashlib
            
            # Create unique cache key based on sequence hash
            seq_hash = hashlib.md5(target_seq.encode()).hexdigest()[:8]
            cache_key = f"{chain_id}_{seq_hash}"
            
            msa_dir = self.output_dir / "msa_cache"
            msa_dir.mkdir(parents=True, exist_ok=True)
            
            # Run MMseqs2 for target chain with unique cache key
            print(f"  🔍 Running MMseqs2 for chain {chain_id} MSA (seq_len={len(target_seq)})...")
            msa_path = process_msa(
                chain_id=cache_key,
                sequence=target_seq,
                msa_dir=msa_dir
            )
            
            if msa_path and Path(msa_path).exists():
                print(f"  ✓ MSA generated: {Path(msa_path).name}")
                return str(msa_path)
            else:
                print(f"  ⚠️  MMseqs2 did not generate MSA")
                return None
        except Exception as e:
            print(f"  ⚠️  MMseqs2 failed: {e}")
            return None
    
    def _build_complex_json(self, binder_seq: str, target_seq: Optional[str], 
                            binder_id: str, seed: int,
                            ligand_ccd: Optional[str] = None,
                            ligand_smiles: Optional[str] = None,
                            nucleic_seq: Optional[str] = None,
                            nucleic_type: str = "dna") -> str:
        """
        Build AF3 input JSON for binder-target complex.
        
        Supports multi-chain targets: if target_seq contains ":" separators,
        each segment becomes a separate chain (B, C, D, etc.).
        
        Supports ligands: if ligand_ccd or ligand_smiles is provided, adds ligand to complex.
        
        Uses validation mode determined by template_path and msa_mode settings:
        - template: Use explicit template file
        - mmseqs: Run MMseqs2 to generate MSA
        - msa_file: Use explicit MSA file
        - single: Sequence-only prediction
        """
        # Handle multi-chain targets (separated by ":")
        if target_seq and ":" in target_seq:
            target_seqs = [s.strip() for s in target_seq.split(":") if s.strip()]
        elif target_seq:
            target_seqs = [target_seq]
        else:
            target_seqs = []
        
        # Chain IDs for targets: B, C, D, E, ...
        target_chain_ids = [chr(ord('B') + i) for i in range(len(target_seqs))]
        
        # Determine next available chain ID for ligand/nucleic (if needed)
        next_chain_id = chr(ord('B') + len(target_seqs))
        
        if target_seqs:
            print(f"  📊 Target chains: {len(target_seqs)} ({', '.join(target_chain_ids)})")
        elif ligand_ccd or ligand_smiles or nucleic_seq:
            print(f"  📊 No target protein chains (ligand/nucleic-only complex)")
        
        # Build sequences list starting with binder (chain A)
        # Binder chain A. By default single-sequence (correct for de novo designs).
        # With msa_binder=True, generate an MSA for the binder too — this matches
        # AF3's default behavior (it MSA-searches every protein chain) and is the
        # right setting for antibodies, whose framework has real evolutionary homologs.
        binder_unpaired_msa = ""
        if self.msa_binder:
            try:
                # Cache the framework MSA once per run and reuse it for every design via
                # query-line swap (germinal-style). Antibody designs in a run share the
                # same framework, so the homolog rows are identical — only the query
                # (CDRs) changes. This avoids one mmseqs call per design.
                if getattr(self, "_binder_msa_homologs", None) is None:
                    self._binder_msa_homologs = ""   # sentinel: attempted
                    self._binder_msa_len = None
                    binder_msa_path = self._run_mmseqs(binder_seq, chain_id="A")
                    if binder_msa_path and Path(binder_msa_path).exists():
                        recs = Path(binder_msa_path).read_text().split("\n>")[1:]  # skip query
                        homolog_recs = []
                        # Cap depth: antibody framework signal saturates well before a few
                        # thousand sequences, and a 10k+ MSA blows up af3 memory (OOM).
                        MAX_BINDER_MSA = 4096
                        for rec in recs:
                            if len(homolog_recs) >= MAX_BINDER_MSA:
                                break
                            parts = rec.split("\n", 1)
                            if len(parts) != 2:
                                continue
                            header, body = parts
                            # strip lowercase insertions → uniform length == query length
                            seq = "".join(c for c in body if not c.islower() and c not in "\n")
                            homolog_recs.append(f">{header}\n{seq}")
                        self._binder_msa_homologs = "\n".join(homolog_recs)
                        self._binder_msa_len = len(binder_seq)
                        print(f"  📄 Binder framework MSA cached: {len(homolog_recs)} homologs "
                              f"(reused across designs via query-swap)")

                if self._binder_msa_homologs and self._binder_msa_len == len(binder_seq):
                    binder_unpaired_msa = f">query\n{binder_seq}\n{self._binder_msa_homologs}"
                    print(f"  📄 Using cached framework MSA for binder chain A ({len(binder_seq)} aa)")
                else:
                    # length mismatch → fall back to a fresh MSA for this design
                    binder_msa_path = self._run_mmseqs(binder_seq, chain_id="A")
                    if binder_msa_path and Path(binder_msa_path).exists():
                        binder_unpaired_msa = Path(binder_msa_path).read_text()
            except Exception as e:
                print(f"  ⚠️  Could not generate binder MSA (falling back to single-seq): {e}")

        # Antibody framework template for chain A (RFdiffusion-style validation):
        # anchor the antibody fold with a structural template instead of/alongside an MSA.
        # Literature standard for designed CDRs (de novo CDRs make MSA of limited utility).
        binder_templates = []
        if self.binder_template_path:
            try:
                b_cif = prepare_template_for_af3(
                    self.binder_template_path,
                    output_dir=self.output_dir / "binder_template_cifs",
                )
                if b_cif.exists():
                    tmpl_len = len(binder_seq)
                    idx = list(range(tmpl_len))
                    binder_templates = [{
                        "mmcifPath": str(b_cif.resolve()),
                        "queryIndices": idx,
                        "templateIndices": idx,
                    }]
                    print(f"  📄 Using framework template for binder chain A: {b_cif.name}")
            except Exception as e:
                print(f"  ⚠️  Could not prepare binder template (skipping): {e}")

        sequences = [
            {
                "protein": {
                    "id": "A",
                    "sequence": binder_seq,
                    "unpairedMsa": binder_unpaired_msa,
                    "pairedMsa": "",
                    "templates": binder_templates,
                }
            }
        ]
        
        # Add each target chain
        for i, (chain_id, seq) in enumerate(zip(target_chain_ids, target_seqs)):
            target_templates = []
            target_unpaired_msa = ""
            
            # Only apply template/MSA to first target chain for simplicity
            # (or could be extended to handle per-chain templates)
            if i == 0:
                if self.validation_mode == "template" and self.template_path:
                    try:
                        # Prepare template (convert PDB to CIF if needed, ensure data_ field exists)
                        template_cif_path = prepare_template_for_af3(
                            self.template_path,
                            output_dir=self.output_dir / "template_cifs"
                        )

                        if template_cif_path.exists():
                            # Override seq with the sequence extracted from the template chain so
                            # that query and template are identical → perfect 1:1 index alignment.
                            # Without this, a naive list(range(len(seq))) mapping is wrong whenever
                            # the template chain has extra residues or is offset from the query.
                            extracted = self._extract_seq_from_template(template_cif_path)
                            if extracted:
                                seq = extracted

                            # Convert to absolute path (required by AF3)
                            template_cif_path_abs = template_cif_path.resolve()

                            seq_len = len(seq)
                            template_indices = list(range(seq_len))
                            target_templates = [{
                                "mmcifPath": str(template_cif_path_abs),
                                "queryIndices": template_indices,
                                "templateIndices": template_indices,
                            }]
                            print(f"  📄 Using template for chain {chain_id}: {template_cif_path.name}")
                        else:
                            print(f"  ⚠️  Template not found after preparation: {self.template_path}")
                    except Exception as e:
                        print(f"  ⚠️  Error preparing template {self.template_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                elif self.validation_mode == "mmseqs":
                    # Run MMseqs2 to generate MSA for this chain
                    msa_path = self._run_mmseqs(seq, chain_id=chain_id)
                    if msa_path:
                        try:
                            with open(msa_path, 'r') as f:
                                target_unpaired_msa = f.read()
                        except Exception as e:
                            print(f"  ⚠️  Could not read MSA: {e}")
                            
                elif self.validation_mode == "msa_file" and self.msa_mode:
                    # Use explicit MSA file
                    msa_path = Path(self.msa_mode)
                    if msa_path.exists():
                        try:
                            with open(msa_path, 'r') as f:
                                target_unpaired_msa = f.read()
                            print(f"  📄 Using MSA file for chain {chain_id}: {msa_path.name}")
                        except Exception as e:
                            print(f"  ⚠️  Could not read MSA: {e}")
                    else:
                        print(f"  ⚠️  MSA file not found: {self.msa_mode}")
                else:
                    print(f"  📄 Single-sequence mode for chain {chain_id} (no template/MSA)")
            
            sequences.append({
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                    "unpairedMsa": target_unpaired_msa,
                    "pairedMsa": "",
                    "templates": target_templates,
                }
            })
        
        # Add ligand if provided
        if ligand_ccd:
            # Handle multiple CCD codes separated by ":" or ","
            if ":" in ligand_ccd:
                ccd_codes = [c.strip() for c in ligand_ccd.split(":") if c.strip()]
            elif "," in ligand_ccd:
                ccd_codes = [c.strip() for c in ligand_ccd.split(",") if c.strip()]
            else:
                ccd_codes = [ligand_ccd.strip()]
            
            ligand_dict = {
                "ligand": {
                    "id": next_chain_id,
                    "ccdCodes": ccd_codes,
                }
            }
            sequences.append(ligand_dict)
            print(f"  🧪 Adding ligand (chain {next_chain_id}): {', '.join(ccd_codes)}")
        elif ligand_smiles:
            ligand_dict = {
                "ligand": {
                    "id": next_chain_id,
                    "smiles": ligand_smiles,
                }
            }
            sequences.append(ligand_dict)
            print(f"  🧪 Adding ligand (chain {next_chain_id}): SMILES")
        
        # Add nucleic acid if provided — split on ":" for multi-strand (e.g. dsDNA)
        if nucleic_seq and isinstance(nucleic_seq, str):
            if ligand_ccd or ligand_smiles:
                next_chain_id = chr(ord(next_chain_id) + 1)

            nucleic_strands = [s.strip() for s in nucleic_seq.split(":") if s.strip()]
            for strand in nucleic_strands:
                nucleic_chain: dict = {
                    "id": next_chain_id,
                    "sequence": strand,
                }
                # Setting unpairedMsa to "" signals AF3's DataPipeline to skip
                # nhmmer/Rfam/RNAcentral search (process_rna_chain checks `is not None`).
                # Without this, RNA MSA search takes minutes even for short sequences.
                if nucleic_type == "rna":
                    nucleic_chain["unpairedMsa"] = ""
                nucleic_dict = {nucleic_type: nucleic_chain}
                sequences.append(nucleic_dict)
                print(f"  🧬 Adding {nucleic_type.upper()} (chain {next_chain_id}): {strand}")
                next_chain_id = chr(ord(next_chain_id) + 1)
        
        json_data = [{
            "name": binder_id,
            "sequences": sequences,
            "modelSeeds": [seed],
            "dialect": "alphafold3",
            "version": 1,
        }]
        return json.dumps(json_data)
    
    # ------------------------------------------------------------------
    # Apo structure helpers
    # ------------------------------------------------------------------

    def _build_apo_json(self, binder_seq: str, binder_id: str, seed: int) -> str:
        """Build AF3 input JSON for the binder alone (no target, no ligand)."""
        json_data = [{
            "name": f"{binder_id}_apo",
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": binder_seq,
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            ],
            "modelSeeds": [seed],
            "dialect": "alphafold3",
            "version": 1,
        }]
        return json.dumps(json_data)

    def validate_single(self,
                        target_seq: Optional[str],
                        binder_seq: str,
                        binder_id: str,
                        **kwargs) -> ValidationResult:
        """
        Validate a single binder sequence with AF3.
        
        Validation mode is determined by template_path and msa_mode settings:
        - template: Use explicit template file
        - mmseqs: Run MMseqs2 to generate MSA
        - msa_file: Use explicit MSA file  
        - single: Sequence-only prediction
        """
        if not self._initialized:
            self.initialize()
        
        from alphafold3.data import pipeline
        from .runtime import get_symbols
        _, _, predict_structure, write_outputs, Input = get_symbols(getattr(self, 'af3_root', None))
        
        print(f"\n[AF3] Validating: {binder_id}")
        print(f"  Binder length: {len(binder_seq)}")
        
        # Handle multi-chain targets for length display
        if target_seq:
            if ":" in target_seq:
                target_seqs_display = [s.strip() for s in target_seq.split(":") if s.strip()]
                total_target_len = sum(len(s) for s in target_seqs_display)
                print(f"  Target: {len(target_seqs_display)} chains, total length: {total_target_len}")
            else:
                print(f"  Target length: {len(target_seq)}")
        else:
            print(f"  Target: None (ligand-only complex)")
        print(f"  Mode: {self.validation_mode}")
        
        # Extract ligand and nucleic acid information from kwargs
        ligand_ccd = kwargs.get('ligand_ccd', None)
        ligand_smiles = kwargs.get('ligand_smiles', None)
        nucleic_seq = kwargs.get('nucleic_seq', None)
        nucleic_type = kwargs.get('nucleic_type', 'dna')
        
        # In template mode, the target sequence can be extracted from the CIF itself —
        # allow target_seq=None when a template is provided.
        if target_seq is None and self.validation_mode == "template" and self.template_path:
            try:
                template_cif = prepare_template_for_af3(
                    self.template_path, output_dir=self.output_dir / "template_cifs"
                )
                target_seq = self._extract_seq_from_template(template_cif)
                if target_seq:
                    print(f"  📄 target_seq inferred from template ({len(target_seq)} aa)")
            except Exception as _e:
                print(f"  ⚠️  Could not extract target_seq from template: {_e}")

        # Validate that we have at least target_seq, ligand, or nucleic acid
        if target_seq is None and not ligand_smiles and not ligand_ccd and not nucleic_seq:
            error_msg = f"target_seq is None for binder {binder_id} and no ligand or nucleic acid provided. Cannot validate without target sequence, ligand, or nucleic acid."
            self.logger.error(error_msg) if hasattr(self, 'logger') and self.logger else None
            raise ValueError(error_msg)
        
        # Build input JSON using class-level template/MSA settings
        json_str = self._build_complex_json(
            binder_seq, target_seq, binder_id, self.seed,
            ligand_ccd=ligand_ccd,
            ligand_smiles=ligand_smiles,
            nucleic_seq=nucleic_seq,
            nucleic_type=nucleic_type
        )
        
        # Create fold input
        fold_input = Input.from_json(json_str)
        fold_input = pipeline.DataPipeline(self.data_pipeline_config).process(fold_input)
        
        try:
            # Run prediction
            all_inference_results = predict_structure(
                fold_input=fold_input,
                model_runner=self.predictor,
                buckets=tuple(int(bucket) for bucket in self.buckets),
                conformer_max_iterations=None,
                print_all_paired_species=False,
            )
            
            # Pick the BEST sample across all seeds/diffusion samples rather than the
            # first one. AF3 antibody–antigen docking is stochastic: reading only
            # inference_results[0] systematically underestimates iPTM (a single
            # non-docked sample scores ~0.1 even for real complexes). Rank by AF3's
            # ranking_score, falling back to iptm.
            def _rank(ir):
                m = ir.metadata
                rs = m.get('ranking_score')
                if rs is not None:
                    try:
                        return float(rs)
                    except (TypeError, ValueError):
                        pass
                return float(m.get('iptm', 0.0))

            _candidates = [ir for grp in all_inference_results for ir in grp.inference_results]
            inference_result = max(_candidates, key=_rank)
            print(f"  Selected best of {len(_candidates)} sample(s) by ranking_score")

            # Save structure
            structure_dir = self.output_dir / "af3_structures"
            structure_dir.mkdir(parents=True, exist_ok=True)
            
            write_outputs(
                all_inference_results,
                output_dir=str(structure_dir),
                name=binder_id
            )
            
            # Convert CIF to PDB
            cif_path = structure_dir / f"{binder_id}.cif"
            pdb_path = structure_dir / f"{binder_id}.pdb"
            
            if cif_path.exists():
                try:
                    from .af_utils import cif_to_pdb
                    cif_to_pdb(str(cif_path), str(pdb_path), remove_cif=False)
                except Exception as e:
                    print(f"  Warning: Could not convert CIF to PDB: {e}")
            
            # Extract metrics
            plddt = float(np.mean(inference_result.predicted_structure.atom_b_factor.tolist())) / 100.0
            iptm_global = float(inference_result.metadata.get('iptm', 0.0))
            ptm = float(inference_result.metadata.get('ptm', 0.0))

            # Binder-specific iPTM from per-chain-pair matrix (chain 0 = binder)
            ipsae = None
            additional_metrics = {}
            cp_iptm = inference_result.metadata.get('chain_pair_iptm', None)
            if cp_iptm is not None and cp_iptm.shape[-1] > 1:
                values = [
                    max(float(cp_iptm[0][i]), float(cp_iptm[i][0]))
                    for i in range(1, cp_iptm.shape[-1])
                ]
                iptm = float(np.max(values)) if values else iptm_global
                additional_metrics['iptm_global'] = iptm_global
            else:
                iptm = iptm_global

            # chain_iptm: per-chain cross-chain iPTM for chain A (binder)
            iptm_xchain = inference_result.metadata.get('iptm_xchain', None)
            if iptm_xchain is not None and len(iptm_xchain) > 0:
                additional_metrics['chain_iptm'] = float(iptm_xchain[0])

            # interface_pae_min: min PAE from binder (chain 0) to first partner (chain 1)
            cp_pae_min = inference_result.metadata.get('chain_pair_pae_min', None)
            if cp_pae_min is not None and cp_pae_min.shape[-1] > 1:
                additional_metrics['interface_pae_min'] = float(cp_pae_min[0][1])
            
            # Handle multi-chain targets (separated by ":")
            if target_seq:
                if ":" in target_seq:
                    target_seqs = [s.strip() for s in target_seq.split(":") if s.strip()]
                else:
                    target_seqs = [target_seq]
            else:
                target_seqs = []
            
            # PAE is stored in numerical_data dict, not as direct attribute
            if hasattr(inference_result, 'numerical_data') and 'full_pae' in inference_result.numerical_data:
                try:
                    pae_np = inference_result.numerical_data['full_pae']
                    if pae_np.ndim == 2:
                        pae_np = pae_np[np.newaxis, ...]

                    # Build asym_ids: 0 = binder, 1..N = protein target chains, then nucleic strands
                    binder_len = len(binder_seq)
                    asym_ids = [0] * binder_len
                    chain_counter = 0
                    for seq in target_seqs:
                        chain_counter += 1
                        asym_ids.extend([chain_counter] * len(seq))
                    # Add nucleic acid strands (each strand is its own chain)
                    if nucleic_seq and isinstance(nucleic_seq, str):
                        nucleic_strands = [s.strip() for s in nucleic_seq.split(":") if s.strip()]
                        for strand in nucleic_strands:
                            chain_counter += 1
                            asym_ids.extend([chain_counter] * len(strand))
                    asym_ids = np.array(asym_ids)
                    num_tokens = len(asym_ids)

                    i_min, i_max, i_mean = ipSAE(
                        num_tokens=num_tokens,
                        asym_ids=asym_ids,
                        full_pae=pae_np
                    )

                    # Average ipSAE across all non-binder chains (protein targets + nucleic strands)
                    num_other_chains = chain_counter
                    if i_mean.shape[1] > 1 and num_other_chains > 0:
                        ipsae_min_values = [float(i_min[0, 0, i+1]) for i in range(num_other_chains) if i+1 < i_min.shape[1]]
                        ipsae_max_values = [float(i_max[0, 0, i+1]) for i in range(num_other_chains) if i+1 < i_max.shape[1]]
                        ipsae_mean_values = [float(i_mean[0, 0, i+1]) for i in range(num_other_chains) if i+1 < i_mean.shape[1]]
                        ipsae = float(np.max(ipsae_min_values)) if ipsae_min_values else None
                        additional_metrics['ipsae_min'] = ipsae
                        additional_metrics['ipsae_max'] = float(np.max(ipsae_max_values)) if ipsae_max_values else None
                        additional_metrics['ipsae_mean'] = float(np.max(ipsae_mean_values)) if ipsae_mean_values else None
                    else:
                        ipsae = None

                except Exception as e:
                    print(f"  Warning: Could not calculate ipSAE: {e}")
            
            # Get chain-specific pLDDT if available
            binder_plddt = None
            target_plddt = None
            
            try:
                b_factors = inference_result.predicted_structure.atom_b_factor
                token_chain_ids = inference_result.metadata.get('token_chain_ids', [])
                
                # pLDDT is per-atom, need to map to per-residue
                atom_chain_ids = inference_result.predicted_structure.chain_id
                binder_atoms = atom_chain_ids == 'A'
                
                if binder_atoms.any():
                    binder_plddt = float(np.mean(b_factors[binder_atoms])) / 100.0

                # For multi-chain targets, average pLDDT across all target chains (B, C, D, ...)
                target_chain_ids = [chr(ord('B') + i) for i in range(len(target_seqs))]
                target_plddts = []
                for chain_id in target_chain_ids:
                    chain_mask = atom_chain_ids == chain_id
                    if chain_mask.any():
                        target_plddts.append(float(np.mean(b_factors[chain_mask])) / 100.0)

                if target_plddts:
                    target_plddt = float(np.mean(target_plddts))
            except Exception as e:
                print(f"  Warning: Could not extract per-chain pLDDT: {e}")
            
            structure_path = str(pdb_path) if pdb_path.exists() else str(cif_path)
            
            print(f"  ✓ iPTM: {iptm:.3f}, pLDDT: {plddt:.3f}")
            if ipsae is not None:
                print(f"    ipSAE: {ipsae:.3f}")
            
            # Add validation mode to metrics for CSV output
            additional_metrics['validation_mode'] = self.validation_mode
            
            # ------------------------------------------------------------------
            # Apo structure prediction and binder RMSD calculation
            # Skip if iPTM is clearly below screening threshold (saves ~3.5s/design)
            # ------------------------------------------------------------------
            binder_apo_rmsd_value = None
            if self._skip_apo_below is not None and iptm < self._skip_apo_below:
                print(f"  Skipping apo prediction (iPTM={iptm:.3f} < iptm_threshold={self._skip_apo_below})")
            else:
                try:
                    print(f"\n  Predicting apo structure for {binder_id} (binder only, no target/ligand)...")
                    apo_json_str = self._build_apo_json(binder_seq, binder_id, self.seed)
                    apo_fold_input = Input.from_json(apo_json_str)
                    apo_fold_input = pipeline.DataPipeline(self.data_pipeline_config).process(apo_fold_input)

                    apo_inference_results = predict_structure(
                        fold_input=apo_fold_input,
                        model_runner=self.predictor,
                        buckets=tuple(int(bucket) for bucket in self.buckets),
                        conformer_max_iterations=None,
                        print_all_paired_species=False,
                    )

                    # Save apo structure
                    apo_structure_dir = self.output_dir / "af3_structures_apo"
                    apo_structure_dir.mkdir(parents=True, exist_ok=True)

                    write_outputs(
                        apo_inference_results,
                        output_dir=str(apo_structure_dir),
                        name=f"{binder_id}_apo",
                    )

                    apo_cif_path = apo_structure_dir / f"{binder_id}_apo.cif"
                    apo_pdb_path = apo_structure_dir / f"{binder_id}_apo.pdb"
                    if apo_cif_path.exists():
                        try:
                            from .af_utils import cif_to_pdb
                            cif_to_pdb(str(apo_cif_path), str(apo_pdb_path), remove_cif=False)
                        except Exception as _e:
                            print(f"  Warning: Could not convert apo CIF to PDB: {_e}")

                    apo_structure_path = str(apo_pdb_path) if apo_pdb_path.exists() else str(apo_cif_path)

                    # Compute Cα RMSD between holo chain A and apo structure
                    holo_coords = self._extract_ca_coords(structure_path, chain_id="A")
                    apo_coords  = self._extract_ca_coords(apo_structure_path, chain_id="A")
                    binder_apo_rmsd_value = self._kabsch_rmsd(holo_coords, apo_coords)

                    if binder_apo_rmsd_value is not None:
                        print(f"  Apo Ca RMSD (holo vs apo binder): {binder_apo_rmsd_value:.3f} A")
                    else:
                        print(f"  Could not calculate apo RMSD for {binder_id}")

                    additional_metrics['apo_structure_path'] = apo_structure_path

                except Exception as _apo_err:
                    print(f"  Apo structure prediction failed for {binder_id}: {_apo_err}")
                    import traceback as _tb
                    _tb.print_exc()
            # ------------------------------------------------------------------

            return ValidationResult(
                binder_id=binder_id,
                binder_sequence=binder_seq,
                iptm=iptm,
                plddt=plddt,
                ipsae=ipsae,
                ptm=ptm,
                binder_plddt=binder_plddt,
                target_plddt=target_plddt,
                structure_path=structure_path,
                binder_apo_rmsd=binder_apo_rmsd_value,
                additional_metrics=additional_metrics
            )

        except Exception as e:
            print(f"  ✗ Error validating {binder_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return ValidationResult(
                binder_id=binder_id,
                binder_sequence=binder_seq,
                iptm=None,
                plddt=None,
                structure_path=None,
                additional_metrics={'error': str(e), 'validation_mode': self.validation_mode}
            )
        
        finally:
            gc.collect()
    
    def validate_batch(self, validation_input, iptm_threshold: float = 0.6, **kwargs):
        """Set apo-skip threshold from iptm_threshold, then run base batch loop."""
        self._skip_apo_below = iptm_threshold
        return super().validate_batch(validation_input, iptm_threshold=iptm_threshold, **kwargs)

    def cleanup(self) -> None:
        """Clean up AF3 resources."""
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        gc.collect()
        super().cleanup()


def validate_with_af3(
    input_csv: str = None,
    design_output_dir: str = None,
    target_sequence: str = None,
    output_dir: str = "./af3_validation",
    gpu_device: str = "0",
    template_path: str = None,  # Explicit template path (like design module)
    msa_mode: str = "single",  # "single", "mmseqs", or path to .a3m file
    num_recycles: int = 10,
    num_diffusion_samples: int = 1,
    seed: int = 1,
    model_dir: str = None,
) -> str:
    """
    Convenience function to run AF3 validation.
    
    Args:
        input_csv: Path to CSV with binder sequences
        design_output_dir: Path to design output directory
        target_sequence: Target protein sequence
        output_dir: Where to save validation results
        gpu_device: GPU device ID
        template_path: Path to template structure (PDB/CIF) - like design module
        msa_mode: "single" (no MSA), "mmseqs" (run MMseqs2), or path to .a3m file
        num_recycles: Number of AF3 recycles
        num_diffusion_samples: Number of diffusion samples
        seed: Random seed
        model_dir: Path to AF3 model directory
        
    Returns:
        Path to results CSV
    """
    if target_sequence is None:
        raise ValueError("target_sequence is required")
    
    # Load input
    if input_csv:
        validation_input = ValidationInput.from_csv(
            input_csv,
            target_seq=target_sequence,
        )
    elif design_output_dir:
        validation_input = ValidationInput.from_design_output(
            design_output_dir,
            target_seq=target_sequence,
        )
    else:
        raise ValueError("Either input_csv or design_output_dir must be provided")
    
    # Run validation with explicit template/MSA settings
    validator = AF3Validator(
        output_dir=output_dir,
        device=gpu_device,
        model_dir=model_dir,
        template_path=template_path,
        msa_mode=msa_mode,
        num_recycles=num_recycles,
        num_diffusion_samples=num_diffusion_samples,
        seed=seed,
    )
    
    try:
        results = validator.validate_batch(validation_input)
        return str(Path(output_dir) / "af3_validation_results.csv")
    finally:
        validator.cleanup()


