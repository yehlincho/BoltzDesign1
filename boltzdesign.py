import os
import sys
import argparse
import yaml
import json
import shutil
import numpy as np
import logging
import subprocess
import pandas as pd
from pathlib import Path
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
sys.path.append(f"{os.getcwd()}/boltzdesign")

# --- Select the GPU BEFORE any torch/CUDA import so CUDA_VISIBLE_DEVICES takes effect. ---
# The rest of the code (and boltzdesign_utils) address the chosen GPU as cuda:0, which only
# works if visibility is restricted here, before torch initializes CUDA. This is what makes
# --gpu_id actually pick the physical GPU (previously it always ran on GPU 0).
def _early_gpu_select():
    import argparse as _ap
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--gpu_id", type=int, default=0)
    _gid = _p.parse_known_args()[0].gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gid)
_early_gpu_select()

from boltzdesign_utils import *
from ligandmpnn_utils import *
from alphafold_utils import *
from input_utils import *
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_gpu_environment(gpu_id):
    """Setup GPU environment variables.

    NOTE: we intentionally do NOT set CUDA_VISIBLE_DEVICES here. It used to be set at
    runtime (after torch/CUDA had already initialized via the top-level imports), so it
    had no effect and the process defaulted to physical GPU 0 (cuda:0) regardless of
    --gpu_id. We now select the device explicitly everywhere via f"cuda:{gpu_id}" (see
    below and boltzdesign_utils.py), which is consistent with the AF3 docker subprocess
    that receives the raw physical GPU id.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BoltzDesign: Protein Design Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design binder for DNA target
  python boltzdesign2.py --name 5zmc --target_type dna --target_seq GCCCTTCCGGGTCCCC,CGGGGACCCGGAAGGG --binder_id A
        """,
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of Run (e.g. DNA_binder_design_1"
    )
    parser.add_argument(
        "--target_type",
        type=str,
        choices=["protein", "peptide", "rna", "dna", "small_molecule", "metal"],
        default="protein",
        help="Type of target molecule",
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="",
        help="Path to a local PDB file (if specify use custom pdb, else fetch from RCSB)",
    )
    parser.add_argument(
        "--pdb_target_ids",
        type=str,
        default="",
        help='Target PDB IDs (comma-separated, e.g., "C,D")',
    )
    parser.add_argument(
        "--pdb_motif_id", type=str, default="", help='Motif PDB ID (e.g., "A")'
    )
    # Custom sequence-based input options
    parser.add_argument(
        "--target_seq",
        type=str,
        default="",
        help='''Custom target sequences/structures (comma-separated). Format depends on target_type:
  - protein/peptide: Amino acid sequences (e.g., "MKTAYIAK,ACDEFGHIK")
  - rna/dna: Nucleotide sequences (e.g., "ATCG,GCTA")
  - small_molecule: SMILES strings (e.g., "[O-]C(=O)C(N)CC[S+](C)C")
  - metal: Element symbols (e.g., "ZN,MG")
Use this OR pdb_path/pdb_target_ids, not both.''',
    )
    parser.add_argument(
        "--custom_motif_input",
        type=str,
        default="",
        help='Custom motif sequences (comma-separated, e.g., "LTQEGT")',
    )
    parser.add_argument("--binder_id", type=str, default="A", help="Binder chain ID")
    parser.add_argument(
        "--use_msa",
        type=str2bool,
        default=False,
        help="Use MSA (if False, runs in single-sequence mode)",
    )
    parser.add_argument(
        "--use_template",
        type=str2bool,
        default=False,
        help="Use template (if False, runs in single-sequence mode)",
    )
    parser.add_argument(
        "--cyclic", type=str2bool, default=False, help="Use cyclic design"
    )
    parser.add_argument(
        "--msa_max_seqs", type=int, default=4096, help="Maximum MSA sequences"
    )
    parser.add_argument(
        "--suffix", type=str, default="0", help="Suffix for the output directory"
    )
    parser.add_argument(
        "--motif_scaffolding",
        type=str2bool,
        default=False,
        help="Use motif scaffolding",
    )

    # Replace start_motif_pos and end_motif_pos with:
    parser.add_argument(
        "--motifs", 
        type=str, 
        default="", 
        help='JSON string of motifs: [{"start_pos":0, "end_pos":10}, ...]'
    )
    parser.add_argument(
        "--min_motif_gap", type=int, default=15, help="Minimum gap between motifs"
    )
    parser.add_argument(
        "--fix_motif_gap_to_min", type=str2bool, default=False, help="Fix gaps to minimum"
    )
    parser.add_argument(
        "--fix_motif_pos",
        type=str,
        default="",
        help='Fix motif position (comma-separated, e.g., "1,2,3" or "all" to fix all positions between start and end)',
    )

    # parser.add_argument(
    #     "--start_motif_pos", type=int, default=0, help="Start motif position"
    # )
    # parser.add_argument(
    #     "--end_motif_pos", type=int, default=0, help="End motif position"
    # )
    # parser.add_argument(
    #     "--fix_motif_pos",
    #     type=str,
    #     default="all",
    #     help='Fix motif position (comma-separated, e.g., "1,2,3" or "all" to fix all positions between start and end)',
    # )
    # Modifications
    parser.add_argument(
        "--modifications",
        type=str,
        default="",
        help='Modifications (comma-separated, e.g., "SEP,SEP")',
    )
    parser.add_argument(
        "--modifications_wt",
        type=str,
        default="",
        help='Modifications (comma-separated, e.g., "S,S")',
    )
    parser.add_argument(
        "--modifications_positions",
        type=str,
        default="",
        help="Modification positions (comma-separated, matching order)",
    )
    parser.add_argument(
        "--modification_target",
        type=str,
        default="",
        help='Target ID for modifications (e.g., "A")',
    )

    # Constraints
    parser.add_argument(
        "--constraint_target",
        type=str,
        default="",
        help='Target ID for constraints (e.g., "A")',
    )
    parser.add_argument(
        "--contact_residues",
        type=str,
        default="",
        help='Contact residues for constraints (comma-separated, e.g., "99,100,109")',
    )

    # Design parameters
    parser.add_argument(
        "--length_min", type=int, default=100, help="Minimum binder length"
    )
    parser.add_argument(
        "--length_max", type=int, default=150, help="Maximum binder length"
    )
    parser.add_argument(
        "--optimizer_type",
        type=str,
        choices=["SGD", "AdamW"],
        default="SGD",
        help="Optimizer type",
    )

    # Iteration parameters
    parser.add_argument(
        "--pre_iteration", type=int, default=0, help="Pre-iteration steps"
    )
    parser.add_argument(
        "--soft_iteration", type=int, default=75, help="Soft iteration steps"
    )
    parser.add_argument(
        "--temp_iteration", type=int, default=50, help="Temperature iteration steps"
    )
    parser.add_argument(
        "--hard_iteration", type=int, default=5, help="Hard iteration steps"
    )
    parser.add_argument(
        "--semi_greedy_steps", type=int, default=2, help="Semi-greedy steps"
    )
    parser.add_argument(
        "--recycling_steps", type=int, default=0, help="Recycling steps"
    )

    parser.add_argument(
        "--use_potential", type=str2bool, default=False, help="Use potential"
    )

    # Advanced configuration
    parser.add_argument(
        "--use_default_config",
        type=str2bool,
        default=True,
        help="Use default configuration (recommended)",
    )
    parser.add_argument(
        "--mask_ligand",
        type=str2bool,
        default=False,
        help="Mask target for warm-up stage",
    )
    parser.add_argument(
        "--optimize_contact_per_binder_pos",
        type=str2bool,
        default=False,
        help="Optimize interface contact per binder position",
    )
    parser.add_argument(
        "--increasing_contact_over_itr",
        type=str2bool,
        default=False,
        help="Increase contacts per iteration, starting with no contacts during pre-iteration",
    )
    parser.add_argument(
        "--distogram_only",
        type=str2bool,
        default=True,
        help="Only use distogram for optimization",
    )
    parser.add_argument(
        "--design_algorithm",
        type=str,
        choices=["3stages", "3stages_extra"],
        default="3stages",
        help="Design algorithm",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--learning_rate_pre",
        type=float,
        default=0.1,
        help="Learning rate for pre iterations (warm-up stage)",
    )
    parser.add_argument(
        "--lr_end",
        type=float,
        default=None,
        help="If set, anneal the base learning rate from --learning_rate (start) to this "
        "value (end) over the design, ON TOP of the built-in temp lr decay. High-early / "
        "low-late decouples the exploration phase (high lr -> compact folds) from the "
        "refinement phase (low lr -> higher confidence). None = fixed lr (default).",
    )
    parser.add_argument(
        "--e_soft", type=float, default=0.8, help="Softmax temperature for 3stages"
    )
    parser.add_argument(
        "--e_soft_1",
        type=float,
        default=0.8,
        help="Initial softmax temperature for 3stages_extra",
    )
    parser.add_argument(
        "--e_soft_2",
        type=float,
        default=1.0,
        help="Additional softmax temperature for 3stages_extra",
    )

    # Interaction parameters
    parser.add_argument(
        "--inter_chain_cutoff", type=int, default=20, help="Inter-chain distance cutoff"
    )
    parser.add_argument(
        "--intra_chain_cutoff", type=int, default=14, help="Intra-chain distance cutoff"
    )
    parser.add_argument(
        "--num_inter_contacts",
        type=int,
        default=1,
        help="Number of inter-chain contacts",
    )
    parser.add_argument(
        "--num_intra_contacts",
        type=int,
        default=4,
        help="Number of intra-chain contacts",
    )

    # loss parameters
    parser.add_argument(
        "--con_loss", type=float, default=1.0, help="Contact loss weight"
    )
    parser.add_argument(
        "--i_con_loss", type=float, default=1.0, help="Inter-chain contact loss weight"
    )
    parser.add_argument(
        "--plddt_loss", type=float, default=0.1, help="pLDDT loss weight"
    )
    parser.add_argument("--pae_loss", type=float, default=0.4, help="PAE loss weight")
    parser.add_argument(
        "--i_pae_loss", type=float, default=0.1, help="Inter-chain PAE loss weight"
    )
    parser.add_argument(
        "--rg_loss", type=float, default=0.0, help="Radius of gyration loss weight"
    )
    parser.add_argument(
        "--grad_noise",
        type=float,
        default=0.0,
        help="SGLD-style gradient-noise scale (dimensionless, relative to gradient "
        "magnitude), annealed to 0 over each stage. Injects noise into binder-position "
        "gradients to escape sharp local minima (e.g. boltz2's warm-up helix basin). "
        "0 = off (standard gradient descent). Try 0.3-1.0 for boltz2.",
    )
    parser.add_argument(
        "--omit_aa_types",
        type=str,
        default="C",
        help="Amino acids (one-letter, concatenated) forbidden during design; masked "
        "from the logits with their gradients zeroed. Default 'C' (cysteine). Use e.g. "
        "'CN' to also forbid asparagine, which boltz2 over-uses as a compositional "
        "attractor that drives mode collapse.",
    )
    parser.add_argument(
        "--helix_loss_max", type=float, default=None,
        help="Max helix loss weight. Default is model-specific (boltz1 0.0, boltz2 -0.05); an explicit value overrides."
    )
    parser.add_argument(
        "--helix_loss_min", type=float, default=None,
        help="Min helix loss weight. Default is model-specific (boltz1 -0.3, boltz2 -0.3); an explicit value overrides."
    )

    # LigandMPNN parameters
    parser.add_argument(
        "--num_designs",
        type=int,
        default=2,
        help="Number of designs per PDB for LigandMPNN",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=4,
        help="Cutoff distance for interface residues (Angstroms)",
    )
    parser.add_argument(
        "--i_ptm_cutoff", type=float, default=0.5, help="iPTM cutoff for redesign"
    )
    parser.add_argument(
        "--complex_plddt_cutoff",
        type=float,
        default=0.7,
        help="Complex pLDDT cutoff for high confidence designs",
    )

    # System configuration
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--design_samples", type=int, default=1, help="Number of design samples"
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--high_iptm", type=str2bool, default=True, help="Disable high iPTM designs"
    )
    # Paths
    parser.add_argument(
        "--boltz_checkpoint",
        type=str,
        default=None,
        help="Path to Boltz checkpoint (default: auto-selected based on boltz_model_version)",
    )
    parser.add_argument(
        "--boltz_model_version",
        type=str,
        choices=["boltz1", "boltz2"],
        default="boltz2",
        help="Boltz model version - boltz1 uses boltz1_conf.ckpt weights, boltz2 uses boltz2_conf.ckpt weights",
    )
    parser.add_argument(
        "--ccd_path", type=str, default="~/.boltz/mols", help="Path to CCD file"
    )
    parser.add_argument(
        "--alphafold_dir", type=str, default="~/alphafold3", help="AlphaFold directory"
    )
    parser.add_argument(
        "--af3_docker_name", type=str, default="alphafold3_yc", help="Docker name"
    )
    parser.add_argument(
        "--af3_database_settings",
        type=str,
        default="~/alphafold3/alphafold3_data_save",
        help="AlphaFold3 database settings",
    )
    parser.add_argument(
        "--af3_hmmer_path",
        type=str,
        default="/home/jupyter-yehlin/.conda/envs/alphafold3_venv",
        help="AlphaFold3 hmmer path, required for RNA MSA generation",
    )
    parser.add_argument(
        "--use_msa_for_af3",
        type=str2bool,
        default=False,
        help="Use MSA for AlphaFold3. Enable this flag if the target requires MSA and you are designing based on Boltzdesign 2 Template mode.",
    )
    # Control flags
    parser.add_argument(
        "--run_boltz_design", type=str2bool, default=True, help="Run Boltz design step"
    )
    parser.add_argument(
        "--run_ligandmpnn",
        type=str2bool,
        default=True,
        help="Run LigandMPNN redesign step",
    )
    parser.add_argument(
        "--run_alphafold",
        type=str2bool,
        default=True,
        help="Run AlphaFold validation step",
    )
    parser.add_argument(
        "--fast_validation",
        type=str2bool,
        default=True,
        help="Use ProteinHunter's warm/batched AF3 validator instead of the per-design "
        "Docker AF3 (same AF3 model, ~4-5x faster end-to-end). Auto-falls back to Docker "
        "if the af3 env / ProteinHunter is not found. Set False to force Docker.",
    )
    parser.add_argument(
        "--af3_ph_env_python",
        type=str,
        default="~/ProteinHunter/.conda/envs/af3/bin/python",
        help="Python interpreter of the af3 env (used only with --fast_validation)",
    )
    parser.add_argument(
        "--proteinhunter_root",
        type=str,
        default="~/ProteinHunter",
        help="ProteinHunter repo root (used only with --fast_validation)",
    )
    parser.add_argument(
        "--af3_num_diffusion_samples",
        type=int,
        default=1,
        help="AF3 diffusion samples for --fast_validation (Docker path uses its own setting)",
    )
    parser.add_argument(
        "--run_rosetta",
        type=str2bool,
        default=True,
        help="Run Rosetta energy calculation (protein targets only)",
    )
    parser.add_argument(
        "--redo_boltz_predict",
        type=str2bool,
        default=False,
        help="Redo Boltz prediction",
    )

    ## Visualization
    parser.add_argument(
        "--show_animation", type=str2bool, default=True, help="Show animation"
    )
    parser.add_argument(
        "--save_trajectory", type=str2bool, default=False, help="Save trajectory"
    )
    
    args = parser.parse_args()
    
    # Auto-select checkpoint based on model version if not explicitly provided
    if args.boltz_checkpoint is None:
        if args.boltz_model_version == "boltz1":
            args.boltz_checkpoint = "~/.boltz/boltz1_conf.ckpt"
        else:  # boltz2
            args.boltz_checkpoint = "~/.boltz/boltz2_conf.ckpt"

    # Model-conditional helix_loss defaults (an explicit --helix_loss_* value overrides).
    # boltz1 tips to beta at ~-0.2, so [-0.3, 0] spans helix<->beta (diversity); boltz2 is
    # ~10x less sensitive and stays confident-helical in [-0.3, -0.05]. Rationale + data:
    # docs/helix_loss_model_specific.md
    if args.helix_loss_min is None:
        args.helix_loss_min = -0.3
    if args.helix_loss_max is None:
        args.helix_loss_max = 0.0 if args.boltz_model_version == "boltz1" else -0.05

    return args


class YamlConfig:
    """Configuration class for managing directories"""

    def __init__(self, main_dir: str = None):
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / "inputs"
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / "PDB"
        self.MSA_DIR = self.MAIN_DIR / "MSA"
        self.YAML_DIR = self.MAIN_DIR / "yaml"

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


def load_boltz_model(args, device):
    """Load Boltz model"""
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    boltz_model = get_boltz_model(
        args.boltz_checkpoint, predict_args, device, args.boltz_model_version, grad_enabled=True, no_potentials=not args.use_potential
    )
    boltz_model.train()
    return boltz_model, predict_args


def load_design_config(target_type, work_dir):
    """
    Load design configuration based on target type.
    Modified so that config files are always loaded from the script's directory,
    instead of using work_dir/boltzdesign/configs.
    """
    # Determine the directory where this script (boltzdesign.py) lives:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The configs directory is under script_dir/boltzdesign/configs/
    config_dir = os.path.join(script_dir, "boltzdesign", "configs2")

    if target_type == "small_molecule":
        config_path = os.path.join(config_dir, "default_sm_config.yaml")
    elif target_type == "metal":
        config_path = os.path.join(config_dir, "default_metal_config.yaml")
    elif target_type == "dna" or target_type == "rna":
        config_path = os.path.join(config_dir, "default_na_config.yaml")
    elif target_type == "protein":
        config_path = os.path.join(config_dir, "default_ppi_config.yaml")
    elif target_type == "peptide":
        config_path = os.path.join(config_dir, "default_pep_config.yaml")
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_explicit_args():
    # Get all command-line arguments (excluding the script name)
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            # Handle --arg=value and --arg value
            if "=" in arg:
                explicit_args.add(arg.split("=")[0].lstrip("-").replace("-", "_"))
            else:
                explicit_args.add(arg.lstrip("-").replace("-", "_"))
    return explicit_args


def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    # Always update these basic parameters regardless of use_default_config
    basic_params = {
        "binder_chain": args.binder_id,
        "non_protein_target": args.target_type not in ["protein", "peptide"],
        "pocket_conditioning": bool(args.contact_residues),
    }

    # Update basic parameters
    explicit_args = get_explicit_args()
    config.update(basic_params)

    # For advanced parameters, only update those that are explicitly set by the user
    # (i.e., different from their default values in argparse)
    parser = argparse.ArgumentParser()
    _, defaults = parser.parse_known_args([])  # Get default values

    advanced_params = {
        "mask_ligand": args.mask_ligand,
        "optimize_contact_per_binder_pos": args.optimize_contact_per_binder_pos,
        "distogram_only": args.distogram_only,
        "design_algorithm": args.design_algorithm,
        "learning_rate": args.learning_rate,
        "learning_rate_pre": args.learning_rate_pre,
        "learning_rate_end": args.lr_end,
        "e_soft": args.e_soft,
        "e_soft_1": args.e_soft_1,
        "e_soft_2": args.e_soft_2,
        "length_min": args.length_min,
        "length_max": args.length_max,
        "inter_chain_cutoff": args.inter_chain_cutoff,
        "intra_chain_cutoff": args.intra_chain_cutoff,
        "num_inter_contacts": args.num_inter_contacts,
        "num_intra_contacts": args.num_intra_contacts,
        "helix_loss_max": args.helix_loss_max,
        "helix_loss_min": args.helix_loss_min,
        "optimizer_type": args.optimizer_type,
        "pre_iteration": args.pre_iteration,
        "soft_iteration": args.soft_iteration,
        "temp_iteration": args.temp_iteration,
        "hard_iteration": args.hard_iteration,
        "semi_greedy_steps": args.semi_greedy_steps,
        "msa_max_seqs": args.msa_max_seqs,
        "recycling_steps": args.recycling_steps,
        "motif_scaffolding": args.motif_scaffolding,
        "motifs": args.motifs,
        "fix_motif_pos": args.fix_motif_pos,
        "min_motif_gap": args.min_motif_gap,
        "fix_motif_gap_to_min": args.fix_motif_gap_to_min,
        "grad_noise": args.grad_noise,
        "omit_aa_types": args.omit_aa_types,
    }

    for param_name, param_value in advanced_params.items():
        if param_name in explicit_args:
            print(f"Updating {param_name} to {param_value}")
            config[param_name] = param_value

    # Model-conditional helix_loss default (unless user set --helix_loss_*)
    _is_b1 = args.boltz_model_version == "boltz1"
    if "helix_loss_min" not in explicit_args:
        config["helix_loss_min"] = -0.3
    if "helix_loss_max" not in explicit_args:
        config["helix_loss_max"] = 0.0 if _is_b1 else -0.05
    # pre_iteration default 0 must win over the stale (30) config YAMLs; the loop above
    # only applies it when the user types --pre_iteration.
    if "pre_iteration" not in explicit_args:
        config["pre_iteration"] = args.pre_iteration
    return config


def run_boltz_design_step(args, config, boltz_model, yaml_dir, main_dir, version_name):
    """Run the Boltz design step"""
    print("Starting Boltz design step...")

    loss_scales = {
        "con_loss": args.con_loss,
        "i_con_loss": args.i_con_loss,
        "plddt_loss": args.plddt_loss,
        "pae_loss": args.pae_loss,
        "i_pae_loss": args.i_pae_loss,
        "rg_loss": args.rg_loss,
    }

    boltz_path = shutil.which("boltz")
    if boltz_path is None:
        raise FileNotFoundError(
            "The 'boltz' command was not found in the system PATH."
        )

    run_boltz_design(
        boltz_path=boltz_path,
        boltz_model_version=args.boltz_model_version,
        main_dir=main_dir,
        yaml_dir=os.path.dirname(yaml_dir),
        boltz_model=boltz_model,
        ccd_path=args.ccd_path,
        design_samples=args.design_samples,
        version_name=version_name,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        redo_boltz_predict=args.redo_boltz_predict,
    )

    print("Boltz design step completed!")


def run_ligandmpnn_step(
    args, main_dir, version_name, ligandmpnn_dir, yaml_dir, work_dir
):
    """Run the LigandMPNN redesign step"""
    print("Starting LigandMPNN redesign step...")
    # Setup LigandMPNN config
    yaml_path = f"{work_dir}/LigandMPNN/run_ligandmpnn_logits_config.yaml"
    with open(yaml_path, "r") as f:
        mpnn_config = yaml.safe_load(f)

    for key, value in mpnn_config.items():
        if isinstance(value, str) and "${CWD}" in value:
            mpnn_config[key] = value.replace("${CWD}", work_dir)

    if not Path(mpnn_config["checkpoint_soluble_mpnn"]).exists():
        raise FileNotFoundError("LigandMPNN checkpoint file not found!")

    with open(yaml_path, "w") as f:
        yaml.dump(mpnn_config, f, default_flow_style=False)

    # Setup directories
    boltzdesign_dir = f"{main_dir}/{version_name}/results_final"
    pdb_save_dir = f"{main_dir}/{version_name}/pdb"

    lmpnn_redesigned_dir = os.path.join(ligandmpnn_dir, "01_lmpnn_redesigned")
    lmpnn_redesigned_fa_dir = os.path.join(ligandmpnn_dir, "01_lmpnn_redesigned_fa")
    lmpnn_redesigned_yaml_dir = os.path.join(ligandmpnn_dir, "01_lmpnn_redesigned_yaml")

    os.makedirs(ligandmpnn_dir, exist_ok=True)
    # Convert CIF to PDB and run LigandMPNN
    convert_cif_files_to_pdb(
        boltzdesign_dir,
        pdb_save_dir,
        high_iptm=args.high_iptm,
        i_ptm_cutoff=args.i_ptm_cutoff,
    )

    if not any(f.endswith(".pdb") for f in os.listdir(pdb_save_dir)):
        print("No successful designs from BoltzDesign")
        sys.exit(1)

    run_ligandmpnn_redesign(
        ligandmpnn_dir,
        pdb_save_dir,
        shutil.which("boltz"),
        os.path.dirname(yaml_dir),
        yaml_path,
        top_k=args.num_designs,
        cutoff=args.cutoff,
        non_protein_target=args.target_type not in ["protein", "peptide"],
        binder_chain=args.binder_id,
        target_chains="all",
        out_dir=lmpnn_redesigned_fa_dir,
        lmpnn_yaml_dir=lmpnn_redesigned_yaml_dir,
        results_final_dir=lmpnn_redesigned_dir,
        boltz_model_version=args.boltz_model_version,
    )

    # Filter high confidence designs
    filter_high_confidence_designs(
        args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir
    )

    print("LigandMPNN redesign step completed!")
    return ligandmpnn_dir


def filter_high_confidence_designs(
    args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir
):
    """Filter and save high confidence designs"""
    print("Filtering high confidence designs...")

    yaml_dir_success_designs_dir = os.path.join(
        ligandmpnn_dir, "01_lmpnn_redesigned_high_iptm"
    )
    yaml_dir_success_boltz_yaml = os.path.join(yaml_dir_success_designs_dir, "yaml")
    yaml_dir_success_boltz_cif = os.path.join(yaml_dir_success_designs_dir, "cif")

    os.makedirs(yaml_dir_success_boltz_yaml, exist_ok=True)
    os.makedirs(yaml_dir_success_boltz_cif, exist_ok=True)

    successful_designs = 0

    # Process designs
    for root in os.listdir(lmpnn_redesigned_dir):
        root_path = os.path.join(lmpnn_redesigned_dir, root, "predictions")
        if not os.path.isdir(root_path):
            continue

        for subdir in os.listdir(root_path):
            json_path = os.path.join(
                root_path, subdir, f"confidence_{subdir}_model_0.json"
            )
            yaml_path = os.path.join(lmpnn_redesigned_yaml_dir, f"{subdir}.yaml")
            cif_path = os.path.join(
                lmpnn_redesigned_dir,
                f"boltz_results_{subdir}",
                "predictions",
                subdir,
                f"{subdir}_model_0.cif",
            )

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                design_name = json_path.split("/")[-2]
                length = int(subdir[subdir.find("length") + 6 : subdir.find("_model")])
                iptm = data.get("iptm", 0)
                complex_plddt = data.get("complex_plddt", 0)

                print(
                    f"{design_name} length: {length} complex_plddt: {complex_plddt:.2f} iptm: {iptm:.2f}"
                )

                if (
                    iptm > args.i_ptm_cutoff
                    and complex_plddt > args.complex_plddt_cutoff
                ):
                    shutil.copy(
                        yaml_path,
                        os.path.join(yaml_dir_success_boltz_yaml, f"{subdir}.yaml"),
                    )
                    shutil.copy(
                        cif_path,
                        os.path.join(yaml_dir_success_boltz_cif, f"{subdir}.cif"),
                    )
                    print(f"✅ {design_name} copied")
                    successful_designs += 1

            except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Skipping {subdir}: {e}")
                continue

    if successful_designs == 0:
        print(
            "Error: No LigandMPNN/ProteinMPNN redesigned designs passed the confidence thresholds"
        )
        sys.exit(1)


def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """Calculate RMSD between holo and apo structures and update confidence CSV.

    Args:
        af_pdb_dir (str): Directory containing holo PDB files
        af_pdb_dir_apo (str): Directory containing apo PDB files
    """
    confidence_csv_path = af_pdb_dir + "/high_iptm_confidence_scores.csv"
    if os.path.exists(confidence_csv_path):
        df_confidence_csv = pd.read_csv(confidence_csv_path)
        for pdb_name in os.listdir(af_pdb_dir):
            if pdb_name.endswith(".pdb"):
                pdb_path = os.path.join(af_pdb_dir, pdb_name)
                pdb_path_apo = os.path.join(af_pdb_dir_apo, pdb_name)
                xyz_holo, _ = get_CA_and_sequence(pdb_path, chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(pdb_path_apo, chain_id="A")
                rmsd = np_rmsd(np.array(xyz_holo), np.array(xyz_apo))
                df_confidence_csv.loc[
                    df_confidence_csv["file"] == pdb_name.split(".pdb")[0] + ".cif",
                    "rmsd",
                ] = rmsd
                print(f"{pdb_path} rmsd: {rmsd}")
        df_confidence_csv.to_csv(confidence_csv_path, index=False)


def run_alphafold_step(args, ligandmpnn_dir, work_dir, mod_to_wt_aa):
    """Run AlphaFold validation step"""
    print("Starting AlphaFold validation step...")

    alphafold_dir = os.path.expanduser(args.alphafold_dir)
    afdb_dir = os.path.expanduser(args.af3_database_settings)
    hmmer_path = os.path.expanduser(args.af3_hmmer_path)
    print("alphafold_dir", alphafold_dir)
    print("afdb_dir", afdb_dir)
    print("hmmer_path", hmmer_path)

    # Create AlphaFold directories
    af_input_dir = f"{ligandmpnn_dir}/02_design_json_af3"
    af_output_dir = f"{ligandmpnn_dir}/02_design_final_af3"
    af_input_apo_dir = f"{ligandmpnn_dir}/02_design_json_af3_apo"
    af_output_apo_dir = f"{ligandmpnn_dir}/02_design_final_af3_apo"

    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Process YAML files
    yaml_dir_success_boltz_yaml = os.path.join(
        ligandmpnn_dir, "01_lmpnn_redesigned_high_iptm", "yaml"
    )

    process_yaml_files(
        yaml_dir_success_boltz_yaml,
        af_input_dir,
        af_input_apo_dir,
        target_name=args.name,
        target_type=args.target_type,
        binder_chain=args.binder_id,
        mod_to_wt_aa=mod_to_wt_aa,
        afdb_dir=afdb_dir,
        hmmer_path=hmmer_path,
        use_msa_for_af3=args.use_msa_for_af3,
    )
    # Run AlphaFold on holo state
    subprocess.run(
        [
            f"{work_dir}/boltzdesign/alphafold.sh",
            af_input_dir,
            af_output_dir,
            str(args.gpu_id),
            alphafold_dir,
            args.af3_docker_name,
        ],
        check=True,
    )

    # Run AlphaFold on apo state
    subprocess.run(
        [
            f"{work_dir}/boltzdesign/alphafold.sh",
            af_input_apo_dir,
            af_output_apo_dir,
            str(args.gpu_id),
            alphafold_dir,
            args.af3_docker_name,
        ],
        check=True,
    )

    print("AlphaFold validation step completed!")

    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"

    convert_cif_files_to_pdb(
        af_output_dir, af_pdb_dir, af_dir=True, high_iptm=args.high_iptm
    )
    if not any(f.endswith(".pdb") for f in os.listdir(af_pdb_dir)):
        print("No successful designs from AlphaFold")
        sys.exit(1)
    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, args.binder_id)

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo


def _ph_af3_available(args):
    """True if the af3 env python, ProteinHunter root, and the driver all exist,
    so --fast_validation can run; otherwise the caller falls back to Docker AF3."""
    py = os.path.expanduser(args.af3_ph_env_python)
    ph_root = os.path.expanduser(args.proteinhunter_root)
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "boltzdesign", "ph_af3_driver.py")
    return (os.path.exists(py)
            and os.path.isdir(os.path.join(ph_root, "validation"))
            and os.path.exists(driver))


def run_alphafold_step_ph(args, ligandmpnn_dir, work_dir, mod_to_wt_aa):
    """AF3 validation via ProteinHunter's warm/batched validator (--fast_validation).

    Same AF3 model as the Docker path; the speedup is engineering (model stays warm,
    MSA cached, no per-design container). PH predicts holo AND apo internally and emits
    PDBs + af3_validation_results.csv. We reshape those into the exact dir/file contract
    the shipped path returns, then reuse `calculate_holo_apo_rmsd` for identical RMSD.
    """
    print("Starting AlphaFold validation step (ProteinHunter warm AF3)...")

    yaml_dir_success = os.path.join(
        ligandmpnn_dir, "01_lmpnn_redesigned_high_iptm", "yaml"
    )
    ph_out_dir = os.path.join(ligandmpnn_dir, "02_ph_af3")
    af_output_dir = os.path.join(ph_out_dir, "af3_structures")
    af_output_apo_dir = os.path.join(ph_out_dir, "af3_structures_apo")
    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    for d in (ph_out_dir, af_pdb_dir, af_pdb_dir_apo):
        os.makedirs(d, exist_ok=True)

    ph_root = os.path.expanduser(args.proteinhunter_root)
    driver = os.path.join(work_dir, "boltzdesign", "ph_af3_driver.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = ph_root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            os.path.expanduser(args.af3_ph_env_python), "-u", driver,
            "--yaml_dir", yaml_dir_success,
            "--out_dir", ph_out_dir,
            "--binder_id", args.binder_id,
            "--gpu", str(args.gpu_id),
            "--proteinhunter_root", ph_root,
            "--num_diffusion_samples", str(args.af3_num_diffusion_samples),
        ],
        check=True, env=env,
    )

    # Reshape PH outputs -> shipped downstream contract. Success gate matches the Docker
    # path: global iptm > 0.5 AND plddt > 0.7 (PH plddt is 0-1; interface_pae reported).
    results_csv = os.path.join(ph_out_dir, "af3_validation_results.csv")
    df = pd.read_csv(results_csv)
    iptm_col = "iptm_global" if "iptm_global" in df.columns else "iptm"
    scores = []
    for _, row in df.iterrows():
        name, iptm, plddt = row["binder_id"], row[iptm_col], row["plddt"]
        if not (iptm > 0.5 and plddt > 0.7):
            continue
        holo = os.path.join(af_output_dir, f"{name}.pdb")
        apo = os.path.join(af_output_apo_dir, f"{name}_apo.pdb")
        if not (os.path.exists(holo) and os.path.exists(apo)):
            print(f"  skip {name}: missing holo/apo PDB")
            continue
        # Match holo/apo by identical filename, as calculate_holo_apo_rmsd expects.
        shutil.copy(holo, os.path.join(af_pdb_dir, f"{name}.pdb"))
        shutil.copy(apo, os.path.join(af_pdb_dir_apo, f"{name}.pdb"))
        scores.append({"file": f"{name}.cif", "iptm": iptm, "plddt": plddt * 100})

    if not scores:
        print("No successful designs from AlphaFold")
        sys.exit(1)
    pd.DataFrame(scores).to_csv(
        os.path.join(af_pdb_dir, "high_iptm_confidence_scores.csv"), index=False
    )
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, args.binder_id)
    print("AlphaFold validation step (ProteinHunter warm AF3) completed!")
    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo


def run_rosetta_step(args, ligandmpnn_dir, af_pdb_dir, af_pdb_dir_apo):
    """Run Rosetta energy calculation (protein targets only)"""
    if args.target_type not in ["protein", "peptide"]:
        print("Skipping Rosetta step (not a protein/peptide target)")
        return

    print("Starting Rosetta energy calculation...")
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"
    from pyrosetta_utils import measure_rosetta_energy

    measure_rosetta_energy(
        af_pdb_dir,
        af_pdb_dir_apo,
        af_pdb_rosetta_success_dir,
        binder_holo_chain=args.binder_id,
        binder_apo_chain="A",
        target=args.target_type,
    )

    print("Rosetta energy calculation completed!")


def setup_environment():
    """Setup environment and parse arguments"""
    args = parse_arguments()
    if args.motifs:
        args.motifs = json.loads(args.motifs)
    if args.fix_motif_pos:
        args.fix_motif_pos = json.loads(args.fix_motif_pos)
    print("motifs", args.motifs)
    work_dir = args.work_dir or os.getcwd()
    os.chdir(work_dir)
    setup_gpu_environment(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args


def assign_chain_ids(target_ids_list, binder_chain="A"):
    """Maps target IDs to unique chain IDs, skipping binder_chain."""
    letters = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != binder_chain]
    return {id: letters[i] for i, id in enumerate(target_ids_list)}


def get_pdb_path(args, config_obj):
    """Get PDB path - try local path first, then RCSB download, otherwise None for single sequence mode"""
    # If local PDB path provided, use it
    if len(args.pdb_path) == 4:
        try:
            print(f"Attempting to download PDB {args.pdb_path} from RCSB...")
            download_pdb(args.pdb_path, config_obj.PDB_DIR)
            pdb_path = config_obj.PDB_DIR / f"{args.pdb_path}.pdb"
            if pdb_path.is_file():
                print(f"Successfully downloaded PDB from RCSB: {pdb_path}")
                return pdb_path
        except Exception as e:
            print(f"Failed to download from RCSB: {e}")

    elif args.pdb_path and Path(args.pdb_path).exists():
        pdb_path = Path(args.pdb_path)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Path exists but is not a file: {args.pdb_path}")
        print(f"Using local PDB: {pdb_path}")
        return pdb_path
    else:
        print("No PDB file available - using no Template mode")
        return None

def initialize_pipeline(args):
    """Initialize models and configurations"""
    work_dir = args.work_dir or os.getcwd()
    boltz_model, _ = load_boltz_model(
        args, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    config_obj = YamlConfig(
        main_dir=f"{work_dir}/inputs/{args.target_type}_{args.name}_{args.suffix}"
    )
    config_obj.setup_directories()
    return boltz_model, config_obj


def generate_yaml_config(args, config_obj):
    """Generate YAML configuration based on input type"""
    # Handle constraints and modifications
    smiles = True
    if args.contact_residues or args.modifications:
        target_ids_list = get_target_ids(args)
        target_id_map = assign_chain_ids(target_ids_list, args.binder_id)
        print(f"Mapped target IDs: {list(target_id_map.values())}")
        constraints, modifications = process_design_constraints(
            target_id_map,
            args.modifications,
            args.modifications_positions,
            args.modification_target,
            args.contact_residues,
            args.constraint_target,
            args.binder_id,
        )
    else:
        constraints, modifications = None, None
    
    target = []
    template_path = None
    
    # Get PDB path (if available)
    pdb_path = get_pdb_path(args, config_obj)
    
    if pdb_path:
        # PDB-based mode
        pdb_target_ids = (
            [str(x.strip()) for x in args.pdb_target_ids.split(",")]
            if args.pdb_target_ids
            else None
        )
        
        # Handle motif scaffolding if enabled
        if args.motif_scaffolding:
            chain_sequences = get_chains_sequence(pdb_path)
            print("chain_sequences", chain_sequences)
            motif_chain_id = args.pdb_motif_id
            motif_sequence = chain_sequences[motif_chain_id]
            target.append(motif_sequence)
        
        # Set template if using one
        if args.use_template:
            template_path = str(pdb_path)
        
        # Extract target sequences based on target type
        if args.target_type in ["rna", "dna"]:
            if not pdb_target_ids:
                raise ValueError("pdb_target_ids required for RNA/DNA targets")
            nucleotide_dict = get_nucleotide_from_pdb(pdb_path)
            for target_id in pdb_target_ids:
                target.append(nucleotide_dict[target_id]["seq"])
                
        elif args.target_type == "small_molecule":
            if len(args.target_seq) < 6:
                smiles = False
                target.append(args.target_seq)
            else:
                ligand_dict = get_ligand_from_pdb(args.target_seq)
                for target_mol in args.target_seq.split(","):
                    print(target_mol, ligand_dict.keys())
                    target.append(ligand_dict[target_mol])

        elif args.target_type in ["protein", "peptide"]:
            if not pdb_target_ids:
                raise ValueError("pdb_target_ids required for protein/peptide targets")
            chain_sequences = get_chains_sequence(pdb_path)
            for target_id in pdb_target_ids:
                print(f"Chain {target_id} sequence: {chain_sequences[target_id]}")
                target.append(chain_sequences[target_id])
        else:
            raise ValueError(f"Unsupported target type: {args.target_type}")
    
    else:
        if args.motif_scaffolding:
            if not args.custom_motif_input:
                raise ValueError("custom_motif_input required for motif scaffolding without PDB")
            target.append(args.custom_motif_input)
        else:
            target_inputs = (
                [str(x.strip()) for x in args.target_seq.split(",")]
                if args.target_seq
                else []
            )
            if not target_inputs:
                raise ValueError("target_seq required when no PDB is available")
            target = target_inputs

            if args.target_type in ["small_molecule"]:
                smiles = False if all(len(item) < 6 for item in target_inputs) else True

    return generate_yaml_for_target_binder(
        args.name,
        args.target_type,
        target,
        config=config_obj,
        binder_id=args.binder_id,
        constraints=constraints,
        modifications=modifications["data"] if modifications else None,
        modification_target=modifications["target"] if modifications else None,
        use_msa=args.use_msa,
        template_path=template_path,
        cyclic=args.cyclic,
        motif_scaffolding=args.motif_scaffolding,
        smiles=smiles
    )

def setup_pipeline_config(args):
    """Setup pipeline configuration"""
    work_dir = args.work_dir or os.getcwd()
    config = load_design_config(args.target_type, work_dir)
    return update_config_with_args(config, args)


def setup_output_directories(args):
    """Setup output directories"""
    work_dir = args.work_dir or os.getcwd()
    main_dir = f"{work_dir}/outputs"
    os.makedirs(main_dir, exist_ok=True)
    return {
        "main_dir": main_dir,
        "version": f"{args.target_type}_{args.name}_{args.suffix}",
    }


def modification_to_wt_aa(modifications, modifications_wt):
    """Convert modifications to WT AA"""
    if not modifications:
        return None, None
    mod_to_wt_aa = {}
    for mod, wt in zip(modifications.split(","), modifications_wt.split(",")):
        mod_to_wt_aa[mod] = wt
    return mod_to_wt_aa


def run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir):
    """Run the pipeline steps based on arguments"""
    # Create AlphaFold directories

    results = {
        "ligandmpnn_dir": f"{output_dir['main_dir']}/{output_dir['version']}/ligandmpnn_cutoff_{args.cutoff}"
    }
    results["af_pdb_dir"] = f"{results['ligandmpnn_dir']}/03_af_pdb_success"
    results["af_pdb_dir_apo"] = f"{results['ligandmpnn_dir']}/03_af_pdb_apo"

    if args.run_boltz_design:
        run_boltz_design_step(
            args,
            config,
            boltz_model,
            yaml_dir,
            output_dir["main_dir"],
            output_dir["version"],
        )

    if args.run_ligandmpnn:
        run_ligandmpnn_step(
            args,
            output_dir["main_dir"],
            output_dir["version"],
            results["ligandmpnn_dir"],
            yaml_dir,
            args.work_dir or os.getcwd(),
        )
    if args.run_alphafold:
        mod_to_wt_aa = modification_to_wt_aa(args.modifications, args.modifications_wt)
        use_ph = args.fast_validation and _ph_af3_available(args)
        if args.fast_validation and not use_ph:
            print("fast_validation requested but af3 env / ProteinHunter not found; "
                  "falling back to Docker AF3.")
        (
            results["af_output_dir"],
            results["af_output_apo_dir"],
            results["af_pdb_dir"],
            results["af_pdb_dir_apo"],
        ) = (run_alphafold_step_ph if use_ph else run_alphafold_step)(
            args, results["ligandmpnn_dir"], args.work_dir or os.getcwd(), mod_to_wt_aa
        )

    if args.run_rosetta:
        run_rosetta_step(
            args,
            results["ligandmpnn_dir"],
            results["af_pdb_dir"],
            results["af_pdb_dir_apo"],
        )

    return results


def main():
    """Main function for running the BoltzDesign pipeline"""
    args = setup_environment()
    boltz_model, config_obj = initialize_pipeline(args)
    yaml_dict, yaml_dir = generate_yaml_config(args, config_obj)

    print("Generated YAML configuration:")
    for key, value in yaml_dict.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")

    # Setup pipeline configuration
    config = setup_pipeline_config(args)
    output_dir = setup_output_directories(args)

    # Run pipeline steps
    print("config:")
    items = list(config.items())
    max_key_len = max(len(key) for key, _ in items)
    max_val_len = max(len(str(val)) for _, val in items)

    # Print header
    print("  " + "=" * (max_key_len + max_val_len + 5))

    # Print items in two columns
    for i in range(0, len(items), 2):
        key1, value1 = items[i]
        if i + 1 < len(items):
            key2, value2 = items[i + 1]
            print(
                f"  {key1:<{max_key_len}}: {str(value1):<{max_val_len}}    "
                f"{key2:<{max_key_len}}: {value2}"
            )
        else:
            print(f"  {key1:<{max_key_len}}: {value1}")

    print("  " + "=" * (max_key_len + max_val_len + 5))
    results = run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
