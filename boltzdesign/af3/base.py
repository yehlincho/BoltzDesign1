"""
Base validation module for cross-validation between structure prediction models.

This module provides a unified interface for validating protein designs
using different folding models (AF3, Boltz, Chai-1).
"""

import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import json
import logging
from datetime import datetime


def calculate_radius_of_gyration(structure_path: str, chain_id: str = "A", verbose: bool = False) -> Optional[float]:
    """
    Calculate radius of gyration (Rg) for a specific chain in a PDB/CIF structure.
    
    Args:
        structure_path: Path to PDB or CIF structure file
        chain_id: Chain ID to calculate Rg for (default: "A" for binder)
        verbose: If True, print detailed error messages
        
    Returns:
        Radius of gyration in Angstroms, or None if calculation fails
    """
    try:
        import gemmi
        _USE_GEMMI = True
    except ImportError:
        _USE_GEMMI = False

    try:
        structure_file = Path(structure_path)
        if not structure_file.exists():
            if verbose:
                print(f"  Error: Structure file does not exist: {structure_path}")
            return None

        if not _USE_GEMMI:
            # Biopython fallback for PDB files (no gemmi required)
            try:
                from Bio.PDB import PDBParser, MMCIFParser
                suffix = structure_file.suffix.lower()
                if suffix in (".cif", ".mmcif"):
                    parser = MMCIFParser(QUIET=True)
                else:
                    parser = PDBParser(QUIET=True)
                struct = parser.get_structure("s", str(structure_file))
                coords = [
                    atom.get_vector().get_array()
                    for model in struct
                    for chain in model
                    if chain.id == chain_id
                    for res in chain
                    for atom in res
                    if atom.get_name() == "CA"
                ]
                if len(coords) < 2:
                    return None
                coords = np.array(coords)
                com = np.mean(coords, axis=0)
                rg = float(np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1))))
                if verbose:
                    print(f"  Successfully calculated Rg: {rg:.2f} Å (using {len(coords)} atoms from chain {chain_id})")
                return rg
            except Exception as e:
                if verbose:
                    print(f"  Error: Rg calculation failed (biopython fallback): {e}")
                return None

        # Read structure
        structure = gemmi.read_structure(str(structure_file))
        
        # Find the specified chain
        coords = []
        chain_found = False
        for model in structure:
            for chain in model:
                if chain.name == chain_id:
                    chain_found = True
                    for residue in chain:
                        for atom in residue:
                            # Only use CA atoms for proteins
                            if atom.element.name == 'C' and atom.name == 'CA':
                                pos = atom.pos
                                coords.append([pos.x, pos.y, pos.z])
        
        if not chain_found:
            # Chain not found - try to find any chain and log warning
            available_chains = []
            for model in structure:
                for chain in model:
                    if chain.name not in available_chains:
                        available_chains.append(chain.name)
            
            if verbose:
                print(f"  Warning: Chain '{chain_id}' not found. Available chains: {available_chains}")
            
            # Try to use first available chain as fallback
            if available_chains:
                fallback_chain = available_chains[0]
                if verbose:
                    print(f"  Using fallback chain '{fallback_chain}' instead")
                chain_id = fallback_chain
                for model in structure:
                    for chain in model:
                        if chain.name == chain_id:
                            for residue in chain:
                                for atom in residue:
                                    if atom.element.name == 'C' and atom.name == 'CA':
                                        pos = atom.pos
                                        coords.append([pos.x, pos.y, pos.z])
            else:
                if verbose:
                    print(f"  Error: No chains found in structure")
                return None
        
        if len(coords) == 0:
            # If no CA atoms found, try all heavy atoms
            if verbose:
                print(f"  Warning: No CA atoms found, trying all heavy atoms")
            for model in structure:
                for chain in model:
                    if chain.name == chain_id:
                        for residue in chain:
                            for atom in residue:
                                if atom.element.name not in ['H', 'D']:  # Exclude hydrogens
                                    pos = atom.pos
                                    coords.append([pos.x, pos.y, pos.z])
        
        if len(coords) < 2:
            if verbose:
                print(f"  Error: Insufficient atoms found ({len(coords)} < 2) for Rg calculation")
            return None
        
        coords = np.array(coords)
        
        # Calculate center of mass
        com = np.mean(coords, axis=0)
        
        # Calculate radius of gyration
        # Rg = sqrt(mean((r_i - r_com)^2))
        squared_distances = np.sum((coords - com) ** 2, axis=1)
        rg = np.sqrt(np.mean(squared_distances))
        
        if verbose:
            print(f"  Successfully calculated Rg: {rg:.2f} Å (using {len(coords)} atoms from chain {chain_id})")
        
        return float(rg)
        
    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        error_msg = f"Failed to calculate Rg for {structure_path} (chain {chain_id}): {e}"
        logger.debug(error_msg)
        if verbose:
            print(f"  Error: {error_msg}")
            import traceback
            traceback.print_exc()
        return None


# Standard column names for unified output format
STANDARD_COLUMNS = {
    "id": ["id", "binder_id", "design_id", "run_id", "name"],
    "sequence": ["sequence", "best_sequence", "best_seq", "binder_sequence", "binder_seq"],
    "iptm": ["iptm", "best_iptm", "interface_ptm"],
    "plddt": ["plddt", "best_plddt", "complex_plddt", "binder_plddt"],
    "ipsae": ["ipsae", "ipsae_min", "best_ipsae"],
    "structure_path": ["structure_path", "best_structure_path", "pdb_path", "pdb_filename", "cif_path", "cif_filename"],
    "target_seq": ["target_seq", "target_sequence", "protein_seq"],
    "target_type": ["target_type"],
    "ligand_smiles": ["ligand_smiles", "smiles"],
    "ligand_ccd": ["ligand_ccd", "ccd"],
    "nucleic_seq": ["nucleic_seq", "nucleic_sequence", "dna_seq", "rna_seq"],
    "nucleic_type": ["nucleic_type"],
}


def _normalize_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names from different pipeline outputs to standard format.

    This handles the different naming conventions used by:
    - boltz_ph: binder_seq/sequence, complex_plddt/plddt
    - chai_ph: wide format with cycle_N_sequence columns (picks last non-null cycle)
    - af3_ph: id, sequence, pdb_path
    """
    rename_map = {}

    for standard_name, aliases in STANDARD_COLUMNS.items():
        if standard_name in df.columns:
            continue  # Already has standard name

        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = standard_name
                break

    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  📋 Normalized columns: {rename_map}")

    # Handle Chai's wide format: cycle_0_sequence, cycle_1_sequence, ...
    # Pick the last non-null cycle sequence for each row.
    if "sequence" not in df.columns:
        cycle_seq_cols = sorted(
            [c for c in df.columns if c.startswith("cycle_") and c.endswith("_sequence")],
            key=lambda c: int(c.split("_")[1])
        )
        if cycle_seq_cols:
            df = df.copy()
            df["sequence"] = df[cycle_seq_cols].apply(
                lambda row: next((v for v in reversed(row.tolist()) if pd.notna(v) and v != ""), None),
                axis=1,
            )
            print(f"  📋 Flattened Chai wide format: picked last cycle from {cycle_seq_cols}")

    return df


@dataclass
class ValidationInput:
    """
    Unified input format for validation.
    
    Can be created from:
    - CSV file with sequences
    - Dictionary with sequences
    - Output from any design pipeline
    """
    target_sequence: Optional[str]  # Can be None for ligand-only designs
    binder_sequences: List[str]
    binder_ids: Optional[List[str]] = None
    ligand_smiles: Optional[str] = None
    ligand_ccd: Optional[str] = None
    target_msa_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.binder_ids is None:
            self.binder_ids = [f"binder_{i}" for i in range(len(self.binder_sequences))]
    
    @classmethod
    def from_csv(cls, csv_path: str, 
                 target_seq: str,
                 sequence_col: str = "sequence",
                 id_col: str = "id",
                 ligand_smiles: Optional[str] = None,
                 ligand_ccd: Optional[str] = None,
                 target_msa_path: Optional[str] = None) -> "ValidationInput":
        """Load validation input from CSV file."""
        df = pd.read_csv(csv_path)
        df = _normalize_design_columns(df)

        if sequence_col not in df.columns:
            raise ValueError(f"Column '{sequence_col}' not found in CSV")
        
        df = df.dropna(subset=[sequence_col])
        binder_seqs = df[sequence_col].tolist()
        binder_ids = df[id_col].tolist() if id_col in df.columns else None

        # Auto-extract target_seq, ligand_smiles, ligand_ccd from CSV columns if not provided
        if target_seq is None and "target_seq" in df.columns:
            vals = df["target_seq"].dropna()
            if len(vals) > 0:
                target_seq = vals.iloc[0]
                print(f"  📋 Auto-extracted target_seq from CSV")

        if ligand_smiles is None and "ligand_smiles" in df.columns:
            vals = df["ligand_smiles"].dropna()
            if len(vals) > 0:
                ligand_smiles = vals.iloc[0]
                print(f"  📋 Auto-extracted ligand_smiles from CSV")

        if ligand_ccd is None and "ligand_ccd" in df.columns:
            vals = df["ligand_ccd"].dropna()
            if len(vals) > 0:
                ligand_ccd = vals.iloc[0]
                print(f"  📋 Auto-extracted ligand_ccd from CSV")

        # Extract any additional metadata
        metadata = {}
        for col in df.columns:
            if col not in [sequence_col, id_col]:
                metadata[col] = df[col].tolist()

        return cls(
            target_sequence=target_seq,
            binder_sequences=binder_seqs,
            binder_ids=binder_ids,
            ligand_smiles=ligand_smiles,
            ligand_ccd=ligand_ccd,
            target_msa_path=target_msa_path,
            metadata=metadata
        )
    
    @classmethod
    def from_design_output(cls, 
                           design_output_dir: str,
                           target_seq: str,
                           ligand_smiles: Optional[str] = None,
                           ligand_ccd: Optional[str] = None,
                           target_msa_path: Optional[str] = None) -> "ValidationInput":
        """
        Load validation input from design output directory.
        Looks for summary CSV files from boltz_ph, chai_ph, or af3_ph outputs.
        
        Automatically normalizes column names from different pipeline formats.
        """
        output_path = Path(design_output_dir)
        
        possible_files = [
            "high_quality_designs.csv",
            "summary_high_iptm.csv",
            "summary_high_metrics.csv",
            "summary_all_runs.csv", 
            "summary.csv",
            "folding_summary.csv",
            "high_iptm_summary.csv"
        ]
        
        summary_df = None
        source_file = None
        for fname in possible_files:
            fpath = output_path / fname
            print(f"Trying to load from: {fpath}")
            if fpath.exists():
                summary_df = pd.read_csv(fpath)
                source_file = fname
                break
        
        if summary_df is None:
            raise FileNotFoundError(
                f"No summary CSV found in {design_output_dir}. "
                f"Tried: {possible_files}"
            )
        
        print(f"📄 Loading from: {source_file}")
        
        # Normalize column names to standard format
        summary_df = _normalize_design_columns(summary_df)
        
        # Now we can use standard column names
        if "sequence" not in summary_df.columns:
            raise ValueError(
                f"No sequence column found in {source_file}. "
                f"Available columns: {list(summary_df.columns)}"
            )
        
        # Filter out empty sequences
        summary_df = summary_df.dropna(subset=["sequence"])
        
        binder_seqs = summary_df["sequence"].tolist()
        binder_ids = summary_df["id"].tolist() if "id" in summary_df.columns else None
        
        if target_seq is None and "target_seq" in summary_df.columns:
            target_vals = summary_df["target_seq"].dropna()
            if len(target_vals) > 0:
                target_seq = target_vals.iloc[0]
                print(f"  📋 Auto-extracted target_seq from CSV: {target_seq[:50]}..." if len(str(target_seq)) > 50 else f"  📋 Auto-extracted target_seq from CSV: {target_seq}")
        
        # Validate that target_seq is not None (unless ligand or nucleic acid is provided)
        # Note: For ligand-only or nucleic-only designs, target_seq can be None if:
        #   - ligand_ccd or ligand_smiles is provided, OR
        #   - nucleic_seq is provided
        # This validation will be done later in the validator if needed
        
        if ligand_smiles is None and "ligand_smiles" in summary_df.columns:
            ligand_vals = summary_df["ligand_smiles"].dropna()
            if len(ligand_vals) > 0:
                ligand_smiles = ligand_vals.iloc[0]
                print(f"  📋 Auto-extracted ligand_smiles from CSV: {ligand_smiles}")
        
        if ligand_ccd is None and "ligand_ccd" in summary_df.columns:
            ccd_vals = summary_df["ligand_ccd"].dropna()
            if len(ccd_vals) > 0:
                ligand_ccd = ccd_vals.iloc[0]
                print(f"  📋 Auto-extracted ligand_ccd from CSV: {ligand_ccd}")
        
    
        nucleic_seq_val = None
        nucleic_type_val = None
        if "nucleic_seq" in summary_df.columns:
            nucleic_vals = summary_df["nucleic_seq"].dropna()
            if len(nucleic_vals) > 0:
                nucleic_seq_val = nucleic_vals.iloc[0]
                print(f"  📋 Auto-extracted nucleic_seq from CSV: {nucleic_seq_val}")
        if "nucleic_type" in summary_df.columns:
            type_vals = summary_df["nucleic_type"].dropna()
            if len(type_vals) > 0:
                nucleic_type_val = type_vals.iloc[0]
        
        # Extract design metadata (metrics from design run)
        metadata = {"source_file": [source_file] * len(binder_seqs)}
        metric_cols = ["iptm", "plddt", "iplddt", "ipsae", "ranking_score", "structure_path"]
        for col in metric_cols:
            if col in summary_df.columns:
                metadata[f"design_{col}"] = summary_df[col].tolist()
        
        if "cycle" in summary_df.columns:
            metadata["design_cycle"] = summary_df["cycle"].tolist()
        
        if nucleic_seq_val:
            metadata["nucleic_seq"] = nucleic_seq_val
            metadata["nucleic_type"] = nucleic_type_val
        
        return cls(
            target_sequence=target_seq,
            binder_sequences=binder_seqs,
            binder_ids=binder_ids,
            ligand_smiles=ligand_smiles,
            ligand_ccd=ligand_ccd,
            target_msa_path=target_msa_path,
            metadata=metadata
        )

    def to_csv(self, output_path: str) -> str:
        """Export validation input to CSV format."""
        data = {
            "id": self.binder_ids,
            "sequence": self.binder_sequences
        }
        data.update(self.metadata)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        return output_path


@dataclass  
class ValidationResult:
    """
    Unified output format for validation results.
    """
    binder_id: str
    binder_sequence: str
    
    iptm: Optional[float] = None
    plddt: Optional[float] = None
    
    # Model-specific metrics
    ipsae: Optional[float] = None  # Chai specific
    ptm: Optional[float] = None
    ranking_score: Optional[float] = None
    
    # Per-chain metrics
    binder_plddt: Optional[float] = None
    target_plddt: Optional[float] = None
    
    # Structure output
    structure_path: Optional[str] = None
    
    # Radius of gyration for binder chain
    binder_rg: Optional[float] = None
    
    # Apo vs holo binder RMSD (binder predicted alone vs in complex)
    binder_apo_rmsd: Optional[float] = None

    # Additional metrics
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "binder_id": self.binder_id,
            "sequence": self.binder_sequence,
            "iptm": self.iptm,
            "plddt": self.plddt,
            "ipsae": self.ipsae,
            "ptm": self.ptm,
            "ranking_score": self.ranking_score,
            "binder_plddt": self.binder_plddt,
            "target_plddt": self.target_plddt,
            "structure_path": self.structure_path,
            "binder_rg": self.binder_rg,
            "binder_apo_rmsd": self.binder_apo_rmsd,
        }
        result.update(self.additional_metrics)
        return result


class BaseValidator(ABC):
    """
    Abstract base class for validation with different folding models.
    """
    
    def __init__(self, 
                 output_dir: str,
                 device: str = "cuda:0",
                 **kwargs):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model = None
        self._initialized = False
        
        # Set up logging to file (model_name will be available after initialization)
        # We'll set up the logger in a method that's called after model_name is available
        self.logger = None
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the validation model."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model. Called lazily before first prediction."""
        pass
    
    @abstractmethod
    def validate_single(self, 
                        target_seq: str,
                        binder_seq: str,
                        binder_id: str,
                        **kwargs) -> ValidationResult:
        """Validate a single binder sequence."""
        pass
    
    def _run_rosetta_relaxation(self, result: ValidationResult, target_seq: Optional[str] = None) -> None:
        """
        Run PyRosetta relaxation and interface scoring on a ValidationResult.
        
        This method is called after filtering, only on designs that passed structure filters.
        It updates the result's additional_metrics with Rosetta scores.
        
        Args:
            result: ValidationResult to relax and score
            target_seq: Target sequence (for multi-chain handling)
        """
        # Default implementation - validators should override this
        # This allows validators to handle structure paths differently
        if not self.include_rosetta_metrics:
            return
        
        try:
            from utils.pyrosetta_utils import pr_relax, score_interface, collapse_multiple_chains
            
            if not result.structure_path or not Path(result.structure_path).exists():
                if self.logger:
                    self.logger.warning(f"  Cannot relax {result.binder_id}: structure file not found")
                return
            
            structure_path = Path(result.structure_path)
            
            # Create relaxed structures directory
            relax_dir = self.output_dir / "relaxed"
            relax_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine relaxed PDB path based on structure path
            if structure_path.suffix.lower() == '.cif':
                # Convert CIF to PDB first
                try:
                    import gemmi
                    structure = gemmi.read_structure(str(structure_path))
                    temp_pdb = relax_dir / f"{result.binder_id}_temp.pdb"
                    structure.write_pdb(str(temp_pdb))
                    pdb_for_rosetta = temp_pdb
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"  Could not convert CIF to PDB for {result.binder_id}: {e}")
                    return
            else:
                pdb_for_rosetta = structure_path
            
            relaxed_pdb = relax_dir / f"{result.binder_id}_{self.model_name}_relaxed.pdb"
            
            # Relax the structure
            print(f"  🔄 Relaxing {result.binder_id} with PyRosetta...")
            pr_relax(str(pdb_for_rosetta), str(relaxed_pdb))
            
            if not relaxed_pdb.exists():
                if self.logger:
                    self.logger.warning(f"  Relaxation failed for {result.binder_id}")
                return
            
            # Handle multi-chain targets
            target_chain_count = 1
            if target_seq and ":" in target_seq:
                target_chain_count = len([s for s in target_seq.split(":") if s.strip()])
            
            if target_chain_count > 1:
                # Collapse all target chains to B for interface analysis
                collapsed_pdb = relax_dir / f"{result.binder_id}_{self.model_name}_collapsed.pdb"
                collapse_multiple_chains(str(relaxed_pdb), str(collapsed_pdb), binder_chain="A", collapse_target="B")
                pdb_for_scoring = str(collapsed_pdb)
            else:
                pdb_for_scoring = str(relaxed_pdb)
            
            # Score interface
            print(f"  📊 Scoring interface for {result.binder_id}...")
            rosetta_metrics, interface_AA, interface_residues = score_interface(
                str(relaxed_pdb), pdb_for_scoring, binder_chain="A", target_chain="B"
            )
            
            # Update result's additional_metrics
            if result.additional_metrics is None:
                result.additional_metrics = {}
            result.additional_metrics.update(rosetta_metrics)
            result.additional_metrics["relaxed_structure_path"] = str(relaxed_pdb)
            result.additional_metrics["interface_residues"] = interface_residues
            
        except Exception as e:
            error_msg = f"PyRosetta relaxation/scoring failed for {result.binder_id}: {e}"
            if self.logger:
                self.logger.warning(error_msg)
            print(f"  ⚠️  {error_msg}")
    
    def validate_batch(self, 
                       validation_input: ValidationInput,
                       iptm_threshold: float = 0.6,
                       plddt_threshold: float = 70.0,
                       apo_rmsd_threshold: Optional[float] = None,
                       **kwargs) -> List[ValidationResult]:
        """
        Validate a batch of binder sequences.
        
        Args:
            validation_input: ValidationInput with target and binder sequences
            iptm_threshold: iPTM threshold for passing validation (default: 0.6)
            plddt_threshold: pLDDT threshold for passing validation (default: 70.0)
            apo_rmsd_threshold: Maximum allowed CA RMSD between apo (binder alone) and holo
                (binder in complex) structures (Å).  None = skip this filter.
            **kwargs: Model-specific parameters
            
        Returns:
            List of ValidationResult objects
        """
        # Set up logger if not already done
        if self.logger is None:
            log_file = self.output_dir / f"{self.model_name}_validation.log"
            # Create a logger specific to this validator instance
            self.logger = logging.getLogger(f"{self.model_name}_validator_{id(self)}")
            self.logger.setLevel(logging.INFO)
            # Remove existing handlers to avoid duplicates
            self.logger.handlers = []
            # Add file handler
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(console_handler)
            self.logger.info(f"Initialized {self.model_name} validator. Log file: {log_file}")
        
        if not self._initialized:
            print(f"Initializing {self.model_name}...")
            self.logger.info(f"Initializing {self.model_name} model...")
            self.initialize()
            self._initialized = True
        
        results = []
        total = len(validation_input.binder_sequences)
        self.logger.info(f"Starting validation of {total} sequences with {self.model_name}")
        
        for idx, (binder_seq, binder_id) in enumerate(
            zip(validation_input.binder_sequences, validation_input.binder_ids)
        ):
            print(f"\n[{self.model_name}] Validating {idx+1}/{total}: {binder_id}")
            self.logger.info(f"Validating {idx+1}/{total}: {binder_id}")
            
            # Extract original structure path from metadata if available
            original_structure_path = None
            if validation_input.metadata and "design_structure_path" in validation_input.metadata:
                structure_paths = validation_input.metadata.get("design_structure_path")
                if structure_paths is not None and isinstance(structure_paths, list) and idx < len(structure_paths) and structure_paths[idx]:
                    original_structure_path = structure_paths[idx]

            # Extract nucleic acid info from metadata (scalar from from_design_output,
            # or per-row list from from_csv)
            nucleic_seq = None
            nucleic_type = "dna"
            if validation_input.metadata:
                ns = validation_input.metadata.get("nucleic_seq")
                if ns is not None:
                    nucleic_seq = ns[idx] if isinstance(ns, list) else ns
                nt = validation_input.metadata.get("nucleic_type")
                if nt is not None:
                    nucleic_type = (nt[idx] if isinstance(nt, list) else nt) or "dna"

            try:
                result = self.validate_single(
                    target_seq=validation_input.target_sequence,
                    binder_seq=binder_seq,
                    binder_id=binder_id,
                    ligand_smiles=validation_input.ligand_smiles,
                    ligand_ccd=validation_input.ligand_ccd,
                    target_msa_path=validation_input.target_msa_path,
                    original_structure_path=original_structure_path,
                    nucleic_seq=nucleic_seq,
                    nucleic_type=nucleic_type,
                    **kwargs
                )
                results.append(result)
                if result.iptm is not None and result.plddt is not None:
                    self.logger.info(f"  Success: {binder_id} - iPTM={result.iptm:.3f}, pLDDT={result.plddt:.3f}")
                else:
                    self.logger.warning(f"  Warning: {binder_id} - Missing metrics (iPTM={result.iptm}, pLDDT={result.plddt})")
                
                # Save intermediate results
                self._save_intermediate_results(results)
                
            except Exception as e:
                error_msg = f"Error validating {binder_id}: {e}"
                print(error_msg)
                self.logger.error(error_msg, exc_info=True)  # Log with full traceback
                results.append(ValidationResult(
                    binder_id=binder_id,
                    binder_sequence=binder_seq,
                    additional_metrics={"error": str(e)}
                ))
        
        # Save final results with validation thresholds
        self._save_final_results(results, validation_input, iptm_threshold, plddt_threshold,
                                 apo_rmsd_threshold=apo_rmsd_threshold)
        
        # Log summary
        successful = sum(1 for r in results if r.iptm is not None and r.plddt is not None)
        failed = len(results) - successful
        self.logger.info(f"Validation complete: {successful} successful, {failed} failed out of {total} total")
        
        return results
    
    def _save_intermediate_results(self, results: List[ValidationResult]) -> None:
        """Save intermediate results to CSV."""
        # Calculate rg for any new results that have structures
        for result in results:
            if result.binder_rg is None:  # Only calculate if not already calculated
                if result.structure_path and Path(result.structure_path).exists():
                    # Use verbose=True to get detailed error messages
                    rg = calculate_radius_of_gyration(result.structure_path, chain_id="A", verbose=True)
                    result.binder_rg = rg
                    if rg is None and self.logger:
                        self.logger.warning(f"  Could not calculate Rg for {result.binder_id} from {result.structure_path}")
                elif result.structure_path:
                    # Structure path exists but file doesn't
                    if self.logger:
                        self.logger.warning(f"  Structure file does not exist for {result.binder_id}: {result.structure_path}")
                    print(f"  ⚠️  Structure file does not exist for {result.binder_id}: {result.structure_path}")
                # If structure_path is None, we can't calculate Rg - that's fine, leave it as None
        
        df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = self.output_dir / f"{self.model_name}_validation_intermediate.csv"
        df.to_csv(csv_path, index=False)
    
    def _save_final_results(self, 
                            results: List[ValidationResult],
                            validation_input: ValidationInput,
                            iptm_threshold: float = 0.6,
                            plddt_threshold: float = 70.0,
                            apo_rmsd_threshold: Optional[float] = None) -> None:
        """Save final results with design metadata and create validated_designs subfolder."""
        # Calculate Rg for any results that don't have it yet (required for filtering)
        # This ensures Rg is calculated before filtering, even if it wasn't done in intermediate results
        for result in results:
            if result.binder_rg is None:  # Only calculate if not already calculated
                if result.structure_path and Path(result.structure_path).exists():
                    # Use verbose=True to get detailed error messages
                    rg = calculate_radius_of_gyration(result.structure_path, chain_id="A", verbose=True)
                    result.binder_rg = rg
                    if rg is None:
                        if self.logger:
                            self.logger.warning(f"  Could not calculate Rg for {result.binder_id} from {result.structure_path}")
                        # Print to console for visibility
                        print(f"  ⚠️  Could not calculate Rg for {result.binder_id} from {result.structure_path}")
                elif result.structure_path:
                    # Structure path exists but file doesn't
                    if self.logger:
                        self.logger.warning(f"  Structure file does not exist for {result.binder_id}: {result.structure_path}")
                    print(f"  ⚠️  Structure file does not exist for {result.binder_id}: {result.structure_path}")
        
        # Create dataframe from results (now includes rg)
        df = pd.DataFrame([r.to_dict() for r in results])
        
        # Add design metadata if available, but never overwrite validation result columns
        if validation_input.metadata:
            validation_cols = set(df.columns)
            for key, values in validation_input.metadata.items():
                if isinstance(values, list) and len(values) == len(results):
                    col_name = f"design_{key}" if key in validation_cols else key
                    df[col_name] = values
        
        # Save full results
        csv_path = self.output_dir / f"{self.model_name}_validation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")
        
        # Create validated_designs subfolder for designs that pass structure module filters only
        validated_dir = self.output_dir / "validated_designs"
        validated_dir.mkdir(parents=True, exist_ok=True)
        
        # Create validated_designs_rosetta subfolder for designs that pass both structure + Rosetta filters
        validated_rosetta_dir = self.output_dir / "validated_designs_rosetta"
        validated_rosetta_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter for designs that pass structure module validation thresholds
        # Requirements: iPTM >= threshold, pLDDT >= threshold, AND Rg < 16.0 Angstroms
        passing_designs = []
        for idx, result in enumerate(results):
            # Check validation metrics
            val_iptm = result.iptm if result.iptm is not None else 0
            val_plddt = result.plddt if result.plddt is not None else 0
            
            # Normalize pLDDT to 0-1 scale if needed (some models output 0-100)
            if val_plddt > 1:
                val_plddt = val_plddt / 100.0
            
            # Normalize threshold to 0-1 scale if needed
            plddt_thresh_normalized = plddt_threshold / 100.0 if plddt_threshold > 1 else plddt_threshold
            
            # Only filter by Rg if it is calculated; otherwise, ignore Rg
            binder_rg = result.binder_rg
            if binder_rg is not None:
                rg_pass = binder_rg < 16.0
            else:
                rg_pass = True  # If Rg is not calculated, ignore it for filtering
            
            # Filter by apo RMSD if threshold is set and value was computed
            if apo_rmsd_threshold is not None and result.binder_apo_rmsd is not None:
                apo_rmsd_pass = result.binder_apo_rmsd < apo_rmsd_threshold
            else:
                apo_rmsd_pass = True  # Skip if not computed or no threshold requested
            
            # Pass if validation metrics meet all thresholds
            if val_iptm >= iptm_threshold and val_plddt >= plddt_thresh_normalized and rg_pass and apo_rmsd_pass:
                passing_designs.append(idx)
        
        # Run PyRosetta relaxation only on designs that passed structure filters (if enabled)
        # This saves time by not relaxing structures that won't pass anyway
        if passing_designs and self.include_rosetta_metrics:
            print(f"\n🔄 Running PyRosetta relaxation on {len(passing_designs)} designs that passed structure filters...")
            for idx, result_idx in enumerate(passing_designs):
                result = results[result_idx]
                if result.structure_path and Path(result.structure_path).exists():
                    print(f"  [{idx+1}/{len(passing_designs)}] Processing {result.binder_id}...")
                    # Run relaxation and scoring on this result
                    self._run_rosetta_relaxation(result, validation_input.target_sequence)
            
            # Update dataframe with new Rosetta metrics
            df = pd.DataFrame([r.to_dict() for r in results])
            if validation_input.metadata:
                for key, values in validation_input.metadata.items():
                    if isinstance(values, list) and len(values) == len(results):
                        df[key] = values
        
        # Save designs that pass structure module filters only
        for idx in passing_designs:
            result = results[idx]
            if result.structure_path and Path(result.structure_path).exists():
                import shutil
                src = Path(result.structure_path)
                dst = validated_dir / src.name
                shutil.copy2(src, dst)
        
        # Normalize threshold for display
        plddt_display = plddt_threshold / 100.0 if plddt_threshold > 1 else plddt_threshold
        
        # Build filter criteria string for display
        filter_str = f"iPTM>={iptm_threshold}, pLDDT>={plddt_display}, Rg<16.0 Å"
        if apo_rmsd_threshold is not None:
            filter_str += f", apo_RMSD<{apo_rmsd_threshold} Å"
        
        # Save validated designs CSV (structure module only)
        if passing_designs:
            validated_df = df.iloc[passing_designs].copy()
            validated_csv = validated_dir / f"{self.model_name}_validated.csv"
            validated_df.to_csv(validated_csv, index=False)
            print(f"✓ {len(passing_designs)} designs passed structure module validation ({filter_str})")
            print(f"✓ Validated designs saved to: {validated_dir}")
        else:
            print(f"⚠ No designs passed structure module validation thresholds ({filter_str})")
        
        # Apply sequence-composition filter (needs only the binder sequence, so
        # it runs even when Rosetta interface metrics were never computed).
        # This catches the over-charged / aromatic-poor "hallucination"
        # signature that passes every confidence metric but fails at the bench.
        composition_passing_designs = []
        if passing_designs and 'sequence' in df.columns:
            validated_composition_dir = self.output_dir / "validated_designs_composition"
            validated_composition_dir.mkdir(parents=True, exist_ok=True)

            passing_df = df.iloc[passing_designs].copy()
            composition_filtered = filter_by_composition(passing_df, add_fail_reasons=True)

            # Carry composition columns back onto the full df for downstream CSVs
            comp_cols = [c for c in composition_filtered.columns if c.startswith('comp_')
                         or c in ('composition_pass', 'composition_fail_reasons')]
            for col in comp_cols:
                df.loc[composition_filtered.index, col] = composition_filtered[col].values

            comp_mask = composition_filtered['composition_pass'] == True
            comp_positions = [i for i, ok in enumerate(comp_mask) if ok]
            composition_passing_designs = [passing_designs[pos] for pos in comp_positions]

            for idx in composition_passing_designs:
                result = results[idx]
                if result.structure_path and Path(result.structure_path).exists():
                    import shutil
                    src = Path(result.structure_path)
                    shutil.copy2(src, validated_composition_dir / src.name)

            if composition_passing_designs:
                comp_validated_df = df.iloc[composition_passing_designs].copy()
                comp_validated_csv = validated_composition_dir / f"{self.model_name}_validated_composition.csv"
                comp_validated_df.to_csv(comp_validated_csv, index=False)
                print(f"✓ {len(composition_passing_designs)} designs passed structure module + composition filters")
                print(f"✓ Composition-validated designs saved to: {validated_composition_dir}")
            else:
                print(f"⚠ No designs passed the sequence-composition filter")

        # Apply Rosetta filter if Rosetta metrics are present
        rosetta_passing_designs = []
        if passing_designs and 'binder_score' in df.columns:
            # Check if Rosetta metrics exist
            rosetta_cols = ['binder_score', 'surface_hydrophobicity', 'interface_sc', 
                           'interface_packstat', 'interface_dG', 'interface_dSASA',
                           'interface_dG_SASA_ratio', 'interface_nres', 
                           'interface_interface_hbonds', 'interface_hbond_percentage',
                           'interface_delta_unsat_hbonds']
            
            if all(col in df.columns for col in rosetta_cols):
                passing_df = df.iloc[passing_designs].copy()
                rosetta_filtered = filter_by_rosetta_metrics(
                    passing_df,
                    binder_score_max=0,
                    surface_hydrophobicity_max=0.35,
                    interface_sc_min=0.55,
                    interface_packstat_min=0,
                    interface_dG_max=0,
                    interface_dSASA_min=1,
                    interface_dG_SASA_ratio_max=0,
                    interface_nres_min=7,
                    interface_hbonds_min=3,
                    interface_hbond_percentage_min=0,
                    interface_delta_unsat_hbonds_max=4,
                    add_fail_reasons=False,  # Don't add reasons to intermediate filtering
                )
                # Get positions of designs that pass Rosetta filter (positions in passing_df)
                rosetta_passing_mask = rosetta_filtered['rosetta_pass'] == True
                rosetta_passing_positions = [i for i, passes in enumerate(rosetta_passing_mask) if passes]
                # Map back to original indices in df
                rosetta_passing_designs = [passing_designs[pos] for pos in rosetta_passing_positions]
                
                # Copy structures for designs that pass both filters
                for idx in rosetta_passing_designs:
                    result = results[idx]
                    if result.structure_path and Path(result.structure_path).exists():
                        import shutil
                        src = Path(result.structure_path)
                        dst = validated_rosetta_dir / src.name
                        shutil.copy2(src, dst)
                
                # Save Rosetta-validated designs CSV
                if rosetta_passing_designs:
                    rosetta_validated_df = df.iloc[rosetta_passing_designs].copy()
                    rosetta_validated_csv = validated_rosetta_dir / f"{self.model_name}_validated_rosetta.csv"
                    rosetta_validated_df.to_csv(rosetta_validated_csv, index=False)
                    print(f"✓ {len(rosetta_passing_designs)} designs passed both structure module + Rosetta filters")
                    print(f"✓ Rosetta-validated designs saved to: {validated_rosetta_dir}")
                else:
                    print(f"⚠ No designs passed Rosetta interface metrics filter")
            else:
                print(f"⚠ Rosetta metrics incomplete - skipping Rosetta filter")
        elif passing_designs:
            print(f"ℹ Rosetta metrics not available - only structure module filter applied")
    
    # ------------------------------------------------------------------
    # Shared apo/holo RMSD helpers (used by every validator sub-class)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_ca_coords(structure_path: str, chain_id: str = "A") -> Optional[np.ndarray]:
        """Extract Cα coordinates for *chain_id* from a PDB or CIF file via gemmi."""
        try:
            import gemmi
            structure = gemmi.read_structure(str(structure_path))
            coords = []
            for model in structure:
                for chain in model:
                    if chain.name == chain_id:
                        for residue in chain:
                            for atom in residue:
                                if atom.name == "CA":
                                    pos = atom.pos
                                    coords.append([pos.x, pos.y, pos.z])
            return np.array(coords, dtype=float) if coords else None
        except Exception as e:
            print(f"  ⚠️  Could not extract CA coords from {structure_path} (chain {chain_id}): {e}")
            return None

    @staticmethod
    def _kabsch_rmsd(P: Optional[np.ndarray], Q: Optional[np.ndarray]) -> Optional[float]:
        """
        Minimum Cα RMSD between two equal-length coordinate arrays P and Q
        via Kabsch superimposition.  Returns None on failure or size mismatch.
        """
        if P is None or Q is None:
            return None
        if len(P) != len(Q) or len(P) == 0:
            print(f"  ⚠️  RMSD: residue count mismatch (holo={len(P)}, apo={len(Q)})")
            return None
        try:
            P_c = P - P.mean(axis=0)
            Q_c = Q - Q.mean(axis=0)
            H = P_c.T @ Q_c
            U, S, Vt = np.linalg.svd(H)
            d = np.linalg.det(Vt.T @ U.T)
            D = np.diag([1.0, 1.0, d])
            R = Vt.T @ D @ U.T
            P_rot = P_c @ R.T
            return float(np.sqrt(np.mean(np.sum((P_rot - Q_c) ** 2, axis=1))))
        except Exception as e:
            print(f"  ⚠️  RMSD calculation failed: {e}")
            return None

    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self._initialized = False
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Amino-acid groupings used by the sequence-composition filter
_CHARGED_AA = set("DEKR")
_AROMATIC_AA = set("FWY")
_HYDROPHOBIC_AA = set("AILMFVWY")


def compute_composition(sequence: str) -> dict:
    """Return sequence-composition descriptors for a binder chain.

    All fractions are over the full binder length. ``net_charge`` is the naive
    (K+R) - (D+E) count, ignoring His and terminal effects (fast, pH-agnostic).
    """
    seq = (sequence or "").strip().upper()
    L = len(seq) or 1
    charged = sum(1 for a in seq if a in _CHARGED_AA)
    aromatic = sum(1 for a in seq if a in _AROMATIC_AA)
    hydrophobic = sum(1 for a in seq if a in _HYDROPHOBIC_AA)
    net_charge = (seq.count("K") + seq.count("R")) - (seq.count("D") + seq.count("E"))
    return {
        "comp_charged_frac": charged / L,
        "comp_aromatic_frac": aromatic / L,
        "comp_hydrophobic_frac": hydrophobic / L,
        "comp_aromatic_count": aromatic,
        "comp_net_charge": net_charge,
    }


def filter_by_composition(
    df: pd.DataFrame,
    charged_frac_max: float = 0.35,
    aromatic_frac_min: float = 0.06,
    aromatic_count_min: int = 2,
    abs_net_charge_max: int = 8,
    sequence_col: str = "sequence",
    add_fail_reasons: bool = True,
) -> pd.DataFrame:
    """Filter designs on binder sequence composition.

    Motivated by the observation that AlphaFold/Boltz-designed binders often
    pass every confidence metric while being over-charged and aromatic-poor --
    the "hallucination" signature that reliably fails experimentally. These
    checks need only the binder sequence, so they apply even when Rosetta
    interface metrics were never computed.

    Args:
        df: DataFrame with a binder ``sequence_col``.
        charged_frac_max: Maximum fraction of D/E/K/R (natural ~0.25).
        aromatic_frac_min: Minimum fraction of F/W/Y (natural ~0.09).
        aromatic_count_min: Minimum absolute number of aromatic residues.
        abs_net_charge_max: Maximum |(K+R)-(D+E)| net charge.
        sequence_col: Column holding the binder sequence.
        add_fail_reasons: Whether to print a failure-reason summary.

    Returns:
        Copy of ``df`` with ``comp_*`` descriptor columns, a boolean
        ``composition_pass`` column, and a ``composition_fail_reasons`` column.
    """
    df = df.copy()

    if sequence_col not in df.columns:
        print(f"Warning: '{sequence_col}' column missing - skipping composition filter.")
        df["composition_pass"] = None
        df["composition_fail_reasons"] = "missing_sequence"
        return df

    comp = df[sequence_col].apply(compute_composition).apply(pd.Series)
    for col in comp.columns:
        df[col] = comp[col].values

    criteria = [
        ("comp_charged_frac", lambda x: x <= charged_frac_max,
         f"charged_frac > {charged_frac_max}"),
        ("comp_aromatic_frac", lambda x: x >= aromatic_frac_min,
         f"aromatic_frac < {aromatic_frac_min}"),
        ("comp_aromatic_count", lambda x: x >= aromatic_count_min,
         f"aromatic_count < {aromatic_count_min}"),
        ("comp_net_charge", lambda x: abs(x) <= abs_net_charge_max,
         f"|net_charge| > {abs_net_charge_max}"),
    ]

    mask = pd.Series(True, index=df.index)
    fail_reasons_list = [[] for _ in range(len(df))]
    for col, criterion, fail_msg in criteria:
        col_mask = df[col].apply(criterion)
        for row_idx, passes in zip(df.index, col_mask):
            if not passes:
                list_idx = df.index.get_loc(row_idx)
                fail_reasons_list[list_idx].append(
                    f"{fail_msg} (actual: {df.loc[row_idx, col]:.2f})")
        mask &= col_mask

    df["composition_pass"] = mask
    df["composition_fail_reasons"] = [
        "; ".join(reasons) if reasons else "" for reasons in fail_reasons_list]

    passed = int(mask.sum())
    total = len(df)
    print(f"Composition filter: {passed}/{total} designs passed "
          f"({passed/total*100:.1f}%)" if total else "Composition filter: 0/0")

    if add_fail_reasons and passed < total:
        all_reasons = []
        for reasons in fail_reasons_list:
            all_reasons.extend(r.split(" (actual:")[0] for r in reasons)
        if all_reasons:
            print("\nComposition failure reason summary:")
            for reason, count in pd.Series(all_reasons).value_counts().items():
                print(f"  {reason}: {count} designs")

    return df


def filter_by_rosetta_metrics(
    df: pd.DataFrame,
    binder_score_max: float = 0,
    surface_hydrophobicity_max: float = 0.35,
    interface_sc_min: float = 0.55,
    interface_packstat_min: float = 0,
    interface_dG_max: float = 0,
    interface_dSASA_min: float = 1,
    interface_dG_SASA_ratio_max: float = 0,
    interface_nres_min: int = 7,
    interface_hbonds_min: int = 3,
    interface_hbond_percentage_min: float = 0,
    interface_delta_unsat_hbonds_max: int = 4,
    add_fail_reasons: bool = True,
) -> pd.DataFrame:
    """
    Filter validation results based on PyRosetta interface metrics.
    
    Args:
        df: DataFrame with validation results including Rosetta metrics
        binder_score_max: Maximum binder energy score (negative is better)
        surface_hydrophobicity_max: Maximum surface hydrophobicity fraction
        interface_sc_min: Minimum interface shape complementarity
        interface_packstat_min: Minimum interface packing statistics
        interface_dG_max: Maximum interface binding energy (negative is better)
        interface_dSASA_min: Minimum interface delta SASA
        interface_dG_SASA_ratio_max: Maximum dG/SASA ratio (negative is better)
        interface_nres_min: Minimum number of interface residues
        interface_hbonds_min: Minimum interface hydrogen bonds
        interface_hbond_percentage_min: Minimum H-bond percentage
        interface_delta_unsat_hbonds_max: Maximum buried unsatisfied H-bonds
        add_fail_reasons: Whether to add a column explaining why each design failed
        
    Returns:
        DataFrame with 'rosetta_pass' column (True/False) and 'rosetta_fail_reasons' column
    """
    # Check if Rosetta metrics are present
    rosetta_cols = ['binder_score', 'surface_hydrophobicity', 'interface_sc', 
                    'interface_packstat', 'interface_dG', 'interface_dSASA',
                    'interface_dG_SASA_ratio', 'interface_nres', 
                    'interface_interface_hbonds', 'interface_hbond_percentage',
                    'interface_delta_unsat_hbonds']
    
    missing_cols = [col for col in rosetta_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing Rosetta columns for filtering: {missing_cols}")
        print("Returning DataFrame with rosetta_pass=None. Run with --include-rosetta-metrics to enable Rosetta scoring.")
        df = df.copy()
        df['rosetta_pass'] = None
        df['rosetta_fail_reasons'] = "missing_rosetta_metrics"
        return df
    
    # Define filter criteria with descriptions
    filter_criteria = [
        ('binder_score', lambda x: x < binder_score_max, f"binder_score >= {binder_score_max}"),
        ('surface_hydrophobicity', lambda x: x < surface_hydrophobicity_max, f"surface_hydrophobicity >= {surface_hydrophobicity_max}"),
        ('interface_sc', lambda x: x > interface_sc_min, f"interface_sc <= {interface_sc_min}"),
        ('interface_packstat', lambda x: x > interface_packstat_min, f"interface_packstat <= {interface_packstat_min}"),
        ('interface_dG', lambda x: x < interface_dG_max, f"interface_dG >= {interface_dG_max}"),
        ('interface_dSASA', lambda x: x > interface_dSASA_min, f"interface_dSASA <= {interface_dSASA_min}"),
        ('interface_dG_SASA_ratio', lambda x: x < interface_dG_SASA_ratio_max, f"interface_dG_SASA_ratio >= {interface_dG_SASA_ratio_max}"),
        ('interface_nres', lambda x: x > interface_nres_min, f"interface_nres <= {interface_nres_min}"),
        ('interface_interface_hbonds', lambda x: x > interface_hbonds_min, f"interface_hbonds <= {interface_hbonds_min}"),
        ('interface_hbond_percentage', lambda x: x > interface_hbond_percentage_min, f"interface_hbond_pct <= {interface_hbond_percentage_min}"),
        ('interface_delta_unsat_hbonds', lambda x: x < interface_delta_unsat_hbonds_max, f"delta_unsat_hbonds >= {interface_delta_unsat_hbonds_max}"),
    ]
    
    df = df.copy()
    mask = pd.Series(True, index=df.index)
    fail_reasons_list = [[] for _ in range(len(df))]
    
    for col, criterion, fail_msg in filter_criteria:
        if col in df.columns:
            col_mask = df[col].apply(criterion)
            # Record fail reasons for rows that fail this criterion
            for idx, (passes, row_idx) in enumerate(zip(col_mask, df.index)):
                if not passes:
                    # Find position in fail_reasons_list
                    list_idx = df.index.get_loc(row_idx)
                    val = df.loc[row_idx, col]
                    fail_reasons_list[list_idx].append(f"{fail_msg} (actual: {val:.2f})")
            mask &= col_mask
    
    # Add columns to DataFrame
    df['rosetta_pass'] = mask
    df['rosetta_fail_reasons'] = ["; ".join(reasons) if reasons else "" for reasons in fail_reasons_list]
    
    passed_count = mask.sum()
    total_count = len(df)
    print(f"Rosetta filter: {passed_count}/{total_count} designs passed ({passed_count/total_count*100:.1f}%)")
    
    # Print summary of failure reasons
    if add_fail_reasons:
        failed_df = df[~mask]
        if len(failed_df) > 0:
            print("\nFailure reason summary:")
            # Count occurrences of each failure type
            all_reasons = []
            for reasons in fail_reasons_list:
                all_reasons.extend([r.split(" (actual:")[0] for r in reasons])
            reason_counts = pd.Series(all_reasons).value_counts()
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count} designs")
    
    return df
