import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import subprocess
from Bio.PDB import MMCIFParser, PDBParser
from pathlib import Path
from IPython.display import HTML, display
import csv
import gc
import shutil
import yaml
import random
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb
from dataclasses import asdict, replace
from boltz.data.types import (
    MSA,
    Coords,
    Ensemble,
    Input,
    Interface,
    StructureV2,
    Structure,
)


CHAIN_TO_NUMBER = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
}


def np_kabsch(a, b, return_v=False):
    """Get alignment matrix for two sets of coordinates using numpy

    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix

    Returns:
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    """
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b

    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)

    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[..., -1] = -u[..., -1]

    return u if return_v else (u @ vh)

def np_rmsd(true, pred):
    """Compute RMSD of coordinates after alignment using numpy

    Args:
        true: Reference coordinates
        pred: Predicted coordinates to align

    Returns:
        Root mean square deviation after optimal alignment
    """
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)

    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)

    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p - q), axis=-1)) + 1e-8)
    
def get_CA_and_sequence(structure_file, chain_id="A"):
    # Determine file type and use appropriate parser
    if structure_file.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    elif structure_file.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("File must be either .cif or .pdb format")

    structure = parser.get_structure("structure", structure_file)
    xyz = []
    sequence = []
    aa_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLU": "E",
        "GLN": "Q",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }

    model = structure[0]  # Get first model (default for most structures)

    if chain_id in model:
        chain = model[chain_id]
        for residue in chain:
            if "CA" in residue:
                xyz.append(residue["CA"].coord)
                sequence.append(aa_map.get(residue.resname, "X"))
    else:
        raise ValueError(f"Chain {chain_id} not found in {structure_file}")

    return xyz, sequence


def align_points(a, b):
    a_centroid = a.mean(axis=0)
    b_centroid = b.mean(axis=0)

    a_centered = a - a_centroid
    b_centered = b - b_centroid

    R = np_kabsch(a_centered, b_centered)
    a_aligned = a_centered @ R + b_centroid
    return a_aligned



def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(pdistogram.device)

    return mid_points


def min_k(x, k=1, mask=None):
    # Convert mask to boolean if it's not None
    if mask is not None:
        mask = mask.bool()  # Convert to boolean tensor

    # Sort the tensor, replacing masked values with Nan
    y = torch.sort(x if mask is None else torch.where(mask, x, float("nan")))[0]

    # Create a mask for the top k value
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    # Compute the mean of the top k values
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def get_con_loss(
    dgram,
    dgram_bins,
    num=None,
    seqsep=None,
    num_pos=float("inf"),
    cutoff=None,
    binary=False,
    mask_1d=None,
    mask_1b=None,
):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:, None] - idx[None, :]
    # Add mask for position separation > 3
    m = (torch.abs(offset) >= seqsep).to(dgram.device)
    if mask_1d is None:
        mask_1d = torch.ones(m.shape[0])
    if mask_1b is None:
        mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    """dgram to contacts"""
    if cutoff is None:
        cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~bins), dim=-1)
    # binary/categorical cross-entropy
    con_loss_cat_ent = -(px_ * torch.log_softmax(dgram, dim=-1)).sum(-1)
    con_loss_bin_ent = -torch.log((bins * px + 1e-8).sum(-1))

    return binary * con_loss_bin_ent + (1 - binary) * con_loss_cat_ent


def mask_loss(x, mask=None, mask_grad=False):
    if mask is None:
        return x.mean()
    else:
        x_masked = (x * mask).sum() / (1e-8 + mask.sum())
        if mask_grad:
            return (x.mean() - x_masked).detach() + x_masked
        else:
            return x_masked


def get_plddt_loss(plddt, mask_1d=None):
    p = 1 - plddt
    return mask_loss(p, mask_1d)


def get_pae_loss(pae, mask_1d=None, mask_1b=None, mask_2d=None):
    pae = pae / 31.0
    L = pae.shape[1]
    if mask_1d is None:
        mask_1d = torch.ones(L).to(pae.device)
    if mask_1b is None:
        mask_1b = torch.ones(L).to(pae.device)
    if mask_2d is None:
        mask_2d = torch.ones((L, L)).to(pae.device)
    mask_2d = mask_2d * mask_1d[:, :, None] * mask_1b[:, None, :]
    return mask_loss(pae, mask_2d)


def get_helix_loss(
    dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs
):
    """helix bias loss"""
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float()
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (
                torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8
            )

    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)


def get_ca_coords(sample_atom_coords, batch, binder_chain="A"):
    atom_to_token = batch["atom_to_token"] * (
        batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain]
    )
    atom_order = torch.cumsum(atom_to_token, dim=1)
    ca_mask = torch.sum((atom_order == 2).to(atom_to_token.dtype), dim=-1)[0]
    ca_coords = sample_atom_coords[:, ca_mask == 1, :]
    return ca_coords


def add_rg_loss(sample_atom_coords, batch, length, binder_chain="A"):
    ca_coords = get_ca_coords(sample_atom_coords, batch, binder_chain)
    center_of_mass = ca_coords.mean(1, keepdim=True)  # keepdim for proper broadcasting
    squared_distances = torch.sum(torch.square(ca_coords - center_of_mass), dim=-1)
    rg = torch.sqrt(squared_distances.mean() + 1e-8)
    rg_th = 2.38 * ca_coords.shape[1] ** 0.365
    loss = torch.nn.functional.elu(rg - rg_th)
    return loss, rg


def aggressive_memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.synchronize()

    for _ in range(3):
        gc.collect()

    torch._dynamo.reset()
    if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
        torch._C._cuda_clearCublasWorkspaces()


def print_memory_usage(stage=""):
    """Print current memory usage for debugging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")



def save_trajectory_pdbs(traj_coords_list, traj_plddt_list, structure, 
                         results_pdb_dir, target_name, itr, length, 
                         boltz_model_version, show_logmd=False):
    """Save trajectory coordinates as PDB files"""
    from logmd import LogMD
    
    if show_logmd:
        logmd = LogMD()
        logmd.notebook()
        print(logmd.url)
    
    atoms = structure.atoms
    ref_coords = traj_coords_list[-1][:atoms["coords"].shape[0], :]
    
    folder_name = f"{target_name}_itr{itr + 1}_length{length}"
    folder_path = Path(results_pdb_dir) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    
    # Save individual frames
    for i, (coords, plddt) in enumerate(zip(traj_coords_list, traj_plddt_list)):
        current_coords = coords[:atoms["coords"].shape[0], :]
        aligned_coords = align_points(current_coords, ref_coords)
        structure.atoms["coords"] = aligned_coords
        structure.atoms["is_present"] = True
        
        pdb_str = to_pdb(structure, plddts=plddt, 
                        boltz2=(boltz_model_version == 'boltz2'))
        
        path = folder_path / f"{i}.pdb"
        path.write_text(pdb_str)
        
        if show_logmd:
            pdb_str_filtered = "\n".join([
                line for line in pdb_str.split("\n")
                if line.startswith("ATOM") or line.startswith("HETATM")
            ])
            logmd(pdb_str_filtered)
    
    # Combine all frames
    combined_path = folder_path / "combined.pdb"
    with combined_path.open("w") as outfile:
        outfile.write((folder_path / "0.pdb").read_text())
        for i in range(1, len(traj_coords_list)):
            outfile.write(f"\n# model {i + 1}\n")
            outfile.write((folder_path / f"{i}.pdb").read_text())


def get_omit_mask(alphabet, omit_aa_types, device):
    """Get mask for omitted amino acid types"""
    omit_indices = [i for i in range(len(alphabet)) if alphabet[i] in omit_aa_types]
    non_protein_indices = [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    return torch.sum(
        torch.eye(33)[omit_indices + non_protein_indices],
        dim=0
    ).to(device) * 1e10


def shift_motifs(motifs, length, fix_motif_pos=None, min_motif_gap=0, fix_motif_gap_to_min=False):
    assert motifs is not None, "motifs are required for motif scaffolding"
    shifted_motifs = []
    shifted_fix_motif_pos = []
    
    total_motif_length = sum(int(m['end_pos']) - int(m['start_pos']) + 1 for m in motifs)
    min_required_gap_space = min_motif_gap * (len(motifs) - 1) if len(motifs) > 1 else 0
    total_gap_space = length - total_motif_length - min_required_gap_space
    
    if total_gap_space < 0:
        raise ValueError(f"Total motif length + gaps exceeds sequence length {length}")
        
    num_gaps = len(motifs) + 1
    gaps = [0] * num_gaps
    remaining_space = total_gap_space
    
    if fix_motif_gap_to_min:
        # Gaps between motifs fixed to min_motif_gap, remainder to ends
        for i in range(1, num_gaps - 1):
            gaps[i] = min_motif_gap
        if num_gaps >= 2:
            first = random.randint(0, remaining_space)
            gaps[0], gaps[-1] = first, remaining_space - first
    else:
        # Random distribution
        temp_gaps = []
        for i in range(num_gaps):
            gap = random.randint(0, remaining_space) if i < num_gaps-1 else remaining_space
            temp_gaps.append(gap)
            remaining_space -= gap
        gaps = temp_gaps
        for i in range(1, len(gaps) - 1):
            gaps[i] += min_motif_gap

    current_pos = gaps[0]
    for i, motif in enumerate(motifs):
        m_len = int(motif['end_pos']) - int(motif['start_pos']) + 1
        shifted_motifs.append({
            **motif,
            'shifted_start': current_pos,
            'shifted_end': current_pos + m_len
        })
        current_pos += m_len + gaps[i+1]
                
    if fix_motif_pos:
        for item in fix_motif_pos:
            idx = int(list(item.keys())[0]) - 1
            pos = int(item[str(idx+1)])
            shifted_pos = pos - shifted_motifs[idx]['start_pos'] + shifted_motifs[idx]['shifted_start']
            shifted_fix_motif_pos.append(shifted_pos)
        return shifted_motifs, shifted_fix_motif_pos
    return shifted_motifs, None

def get_motif_mask(chain_indices, shifted_start, shifted_end, fix_positions=None):
    """Create mask for motif positions"""
    if fix_positions is None:
        return (torch.arange(len(chain_indices)) < shifted_start) | \
               (torch.arange(len(chain_indices)) >= shifted_end)
    else:
        mask = torch.ones(len(chain_indices), dtype=torch.bool)
        mask[fix_positions] = False
        return mask

def update_batch_msa(batch, binder_chain_num, non_protein_target, device):
    """Update MSA and profile in batch"""
    if non_protein_target:
        batch["msa"] = batch["res_type"].unsqueeze(0).to(device).detach()
        batch["profile"] = batch["msa"].float().mean(dim=0).to(device).detach()
    else:
        batch["msa"][:, 0, :, :] = batch["res_type"].to(device).detach()
        binder_mask = batch["entity_id"] == binder_chain_num
        batch["profile"][binder_mask, :] = (
            batch["msa"][:, 0, binder_mask[0], :].float().mean(dim=1).to(device).detach()
        )
    return batch


def extract_sequence_from_batch(batch, binder_chain_num, alphabet):
    """Extract amino acid sequence from batch"""
    return "".join([
        alphabet[i]
        for i in torch.argmax(
            batch["res_type"][batch["entity_id"] == binder_chain_num, :],
            dim=-1,
        ).detach().cpu().numpy()
    ])


def save_confidence_scores(
    folder_dir, output, structure, name, model_idx=0, boltz2=True
):
    output_dir = os.path.join(folder_dir, f"boltz_results_{name}", "predictions", name)

    os.makedirs(output_dir, exist_ok=True)
    atoms = structure.atoms
    atoms["coords"] = (
        output["coords"][0].detach().cpu().numpy()[: atoms["coords"].shape[0], :]
    )
    atoms["is_present"] = True
    residues = structure.residues
    residues["is_present"] = True
    interfaces = np.array([], dtype=Interface)
    if boltz2:
        new_structure: StructureV2 = replace(
            structure,
            atoms=atoms,
            residues=residues,
            interfaces=interfaces,
        )
    else:
        new_structure: Structure = replace(
            structure,
            atoms=atoms,
            residues=residues,
            interfaces=interfaces,
            connections=structure.connections,
        )
    plddts = output["plddt"].detach().cpu().numpy()[0]
    path = Path(output_dir) / f"{name}_model_{model_idx}.cif"
    with path.open("w") as f:
        f.write(to_mmcif(new_structure, plddts=plddts, boltz2=True if boltz2 else False))

    # Save confidence summary
    if "plddt" in output:
        confidence_summary_dict = {}
        for key in [
            "confidence_score",
            "ptm",
            "iptm",
            "ligand_iptm",
            "protein_iptm",
            "complex_plddt",
            "complex_iplddt",
            "complex_pde",
            "complex_ipde",
        ]:
            if key in output:
                confidence_summary_dict[key] = output[key].item()

        if "pair_chains_iptm" in output:
            confidence_summary_dict["chains_ptm"] = {
                idx: output["pair_chains_iptm"][idx][idx].item()
                for idx in output["pair_chains_iptm"]
            }
            confidence_summary_dict["pair_chains_iptm"] = {
                idx1: {
                    idx2: output["pair_chains_iptm"][idx1][idx2].item()
                    for idx2 in output["pair_chains_iptm"][idx1]
                }
                for idx1 in output["pair_chains_iptm"]
            }

        json_path = os.path.join(
            output_dir, f"confidence_{name}_model_{model_idx}.json"
        )
        with open(json_path, "w") as f:
            json.dump(confidence_summary_dict, f, indent=4)
        # Save plddt
        plddt = output["plddt"]
        plddt_path = os.path.join(output_dir, f"plddt_{name}_model_{model_idx}.npz")
        np.savez_compressed(plddt_path, plddt=plddt.cpu().detach().numpy())

    if "pae" in output:
        pae = output["pae"]
        pae_path = os.path.join(output_dir, f"pae_{name}_model_{model_idx}.npz")
        np.savez_compressed(pae_path, pae=pae.cpu().detach().numpy())



def plot_loss_history(loss_history, con_loss_history, i_con_loss_history,
                     loss_dir, target_name, itr, length):
    """Create and save loss history plots"""
    plt.style.use("dark_background")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#1C1C1C")
    
    colors = ["#00ff99", "#ff3366", "#3366ff"]
    plot_configs = [
        (ax1, loss_history, "Total Loss", colors[0]),
        (ax2, con_loss_history, "Intra-Contact Loss", colors[1]),
        (ax3, i_con_loss_history, "Inter-Contact Loss", colors[2]),
    ]
    
    for ax, data, title, color in plot_configs:
        ax.plot(data, color=color, linewidth=2)
        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title} History", fontsize=14, pad=15)
        ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    
    save_path = os.path.join(
        loss_dir,
        f"{target_name}_loss_history_itr{itr + 1}_length{length}.png"
    )
    plt.savefig(save_path, facecolor="#1C1C1C", edgecolor="none",
                bbox_inches="tight", dpi=300)
    plt.show()



def setup_output_directories(version_dir):
    """Create all necessary output directories"""
    directories = {
        'results_final': os.path.join(version_dir, "results_final"),
        'results_yaml': os.path.join(version_dir, "results_yaml"),
        'results_final_apo': os.path.join(version_dir, "results_final_apo"),
        'results_yaml_apo': os.path.join(version_dir, "results_yaml_apo"),
        'loss': os.path.join(version_dir, "loss"),
        'animation': os.path.join(version_dir, "animation"),
        'pdb': os.path.join(version_dir, "pdb"),
    }
    
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories




def save_yaml_configs(yaml_path, best_sequence, binder_chain_num,
                     results_yaml_dir, results_yaml_dir_apo,
                     target_name, itr, length):
    """Save YAML configuration files for holo and apo structures"""
    result_yaml = os.path.join(
        results_yaml_dir,
        f"{target_name}_results_itr{itr + 1}_length{length}.yaml"
    )
    result_yaml_apo = os.path.join(
        results_yaml_dir_apo,
        f"{target_name}_results_itr{itr + 1}_length{length}.yaml"
    )
    
    # Save holo config
    shutil.copy2(yaml_path, result_yaml)
    with open(result_yaml) as f:
        data = yaml.safe_load(f)
    
    data["sequences"][binder_chain_num]["protein"]["sequence"] = best_sequence
    data.pop("constraints", None)
    
    # Convert MSA files from npz to a3m format
    for seq in data["sequences"]:
        if "protein" in seq and "msa" in seq["protein"] and seq["protein"]["msa"]:
            seq["protein"]["msa"] = seq["protein"]["msa"].replace(".npz", ".a3m")
    
    with open(result_yaml, "w") as f:
        yaml.dump(data, f)
    
    # Save apo config
    shutil.copy2(result_yaml, result_yaml_apo)
    with open(result_yaml_apo) as f:
        data_apo = yaml.safe_load(f)
    
    data_apo["sequences"] = [data_apo["sequences"][binder_chain_num]]
    data_apo.pop("constraints", None)
    
    with open(result_yaml_apo, "w") as f:
        yaml.dump(data_apo, f)
    
    return result_yaml, result_yaml_apo



def cleanup_iteration(output, output_apo, best_batch, best_batch_apo,
                     best_structure, best_structure_apo, distogram_history_2,
                     sequence_history_2, loss_history_2, con_loss_history,
                     i_con_loss_history, plddt_loss_history, traj_coords_list_2,
                     traj_plddt_list_2, structure, distogram_history,
                     sequence_history, loss_history, traj_coords_list, traj_plddt_list):
    """Clean up memory after each iteration"""
    # Move tensors to CPU
    output = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output.items()}
    output_apo = {k: v.cpu() if torch.is_tensor(v) else v for k, v in output_apo.items()}
    
    # Clear lists
    traj_coords_list.clear()
    traj_plddt_list.clear()
    
    # Delete large objects
    del (output, output_apo, best_batch, best_batch_apo, best_structure,
         best_structure_apo, distogram_history_2, sequence_history_2,
         loss_history_2, con_loss_history, i_con_loss_history, plddt_loss_history,
         traj_coords_list_2, traj_plddt_list_2, structure, distogram_history,
         sequence_history, loss_history, traj_coords_list, traj_plddt_list)
    
    aggressive_memory_cleanup()





def visualize_training_history(
    best_batch,
    loss_history,
    sequence_history,
    distogram_history,
    length,
    binder_chain="A",
    save_dir=None,
    save_filename=None,
):
    """
    Visualize training history including loss plot, distogram animation, and sequence evolution animation.
    Args:
        loss_history (list): List of loss values over training
        sequence_history (list): List of sequence probability matrices over training
        distogram_history (list): List of distogram matrices over training
        length (int): Length of sequence to visualize
        save_dir (str): Directory to save visualizations
    """

    mask = (
        (best_batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain])
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    sequence_history = [seq[mask] for seq in sequence_history]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    def create_distogram_animation():
        plt.style.use("default")  # Use default white background style
        fig, ax = plt.subplots(figsize=(6, 6))
        distogram_2d = distogram_history[0]
        im = ax.imshow(distogram_2d)

        plt.colorbar(im, ax=ax)
        ax.set_title("Distogram Evolution")

        def update(frame):
            distogram_2d = distogram_history[frame]
            im.set_data(distogram_2d)
            ax.set_title(f"Distogram Epoch {frame + 1}")
            return (im,)
 
        ani = FuncAnimation(fig, update, frames=len(distogram_history), interval=200)
        if save_dir:
            ani.save(
                os.path.join(save_dir, f"{save_filename}_distogram_evolution.gif"),
                writer="pillow",
            )
        plt.close()
        return ani

    # Create sequence evolution animation
    def create_sequence_animation():
        plt.style.use("default")  # Use default white background style
        fig, ax = plt.subplots(figsize=(12, 3.5))
        im = ax.imshow(
            sequence_history[0].T,
            vmin=0,
            vmax=1,
            cmap="Blues",
            aspect="auto",
            alpha=0.8,
        )
        plt.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(list("ARNDCQEGHILKMFPSTWYV"))
        ax.set_title("Sequence Evolution")

        def update(frame):
            im.set_data(sequence_history[frame].T)
            ax.set_title(f"Sequence Epoch {frame + 1}")
            return (im,)

        ani = FuncAnimation(fig, update, frames=len(sequence_history), interval=200)
        if save_dir:
            ani.save(
                os.path.join(save_dir, f"{save_filename}_sequence_evolution.gif"),
                writer="pillow",
            )
        plt.close()
        return ani

    # Create and save animations
    distogram_ani = create_distogram_animation()
    sequence_ani = create_sequence_animation()

    return distogram_ani, sequence_ani


def run_boltz_predictions(boltz_path, boltz_model_version, result_yaml,
                         result_yaml_apo, results_final_dir, results_final_dir_apo):
    """Run Boltz predictions on holo and apo structures"""
    for yaml_file, out_dir in [
        (result_yaml, results_final_dir),
        (result_yaml_apo, results_final_dir_apo)
    ]:
        subprocess.run([
            boltz_path, "predict", str(yaml_file),
            "--out_dir", str(out_dir),
            "--model", boltz_model_version,
            "--write_full_pae",
        ])

def process_design_results(
    output,
    output_apo,
    best_batch,
    best_batch_apo,
    best_structure,
    best_structure_apo,
    distogram_history,
    sequence_history,
    loss_history,
    con_loss_history,
    i_con_loss_history,
    plddt_loss_history,
    traj_coords_list,
    traj_plddt_list,
    structure,
    config,
    directories,
    yaml_path,
    target_name,
    itr,
    loss_scales,
    boltz_path,
    boltz_model_version,
    alphabet,
    redo_boltz_predict=True,
    show_animation=False,
    save_plots=False,
    save_trajectory=False,
):
    """Process and save design results"""
    
    # Print metrics
    print("-" * 100)
    print(f"Holo Protein PLDDT: {output['plddt'][:config['length']].mean():.3f}")
    print(f"Apo Protein PLDDT: {output_apo['plddt'][:config['length']].mean():.3f}")
    print("-" * 100)
    print(f"Holo Complex PLDDT: {float(output['complex_plddt'].detach().cpu().numpy()):.3f}")
    print(f"Apo Complex PLDDT: {float(output_apo['complex_plddt'].detach().cpu().numpy()):.3f}")
    print("-" * 100)
    
    # Calculate RMSD
    ca_coords = get_ca_coords(
        output["coords"], best_batch, binder_chain=config["binder_chain"]
    ).detach().cpu().numpy()
    
    ca_coords_apo = get_ca_coords(
        output_apo["coords"], best_batch_apo, binder_chain="A"
    ).detach().cpu().numpy()
    
    rmsd = np_rmsd(ca_coords, ca_coords_apo)
    print("-" * 100)
    print("RMSD:", rmsd)
    print("-" * 100)
    
    # Save trajectory if requested
    if save_trajectory:
        save_trajectory_pdbs(
            traj_coords_list, traj_plddt_list, structure,
            directories['pdb'], target_name, itr, config['length'],
            boltz_model_version, show_logmd=False
        )
    
    # Plot and save loss history. Building the 125-frame distogram/sequence animations
    # costs ~4.9s per design and nothing reads them on a normal run, so it is opt-in.
    if save_plots or show_animation:
      try:
        plot_loss_history(
            loss_history, con_loss_history, i_con_loss_history,
            directories['loss'], target_name, itr, config['length']
        )
        
        # Create animations
        distogram_ani, sequence_ani = visualize_training_history(
            best_batch, loss_history, sequence_history, distogram_history,
            config['length'], binder_chain=config['binder_chain'],
            save_dir=directories['animation'],
            save_filename=f"{target_name}_itr{itr + 1}_length{config['length']}"
        )
        
        if show_animation:
            display(HTML(
                f"<div style='display:flex;gap:10px'>"
                f"<div style='flex:0.4'>{distogram_ani.to_jshtml()}</div>"
                f"<div style='flex:0.6'>{sequence_ani.to_jshtml()}</div>"
                f"</div>"
            ))
      except Exception as e:
        # previously `return None`, which silently skipped the RMSD csv and the
        # confidence scores below whenever plotting failed
        print(f"Error plotting loss history: {str(e)}")
    
    # Save RMSD to CSV
    rmsd_csv_path = os.path.join(directories['results_final'], "rmsd_results.csv")
    csv_exists = os.path.exists(rmsd_csv_path)
    
    with open(rmsd_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow([
                "target", "length", "iteration", "apo_holo_rmsd",
                "complex_plddt", "iptm", "helix_loss"
            ])
        writer.writerow([
            target_name, config["length"], itr + 1, rmsd,
            output["complex_plddt"].item(), output["iptm"].item(),
            loss_scales["helix_loss"]
        ])
    
    # Extract and save sequence
    binder_chain_num = CHAIN_TO_NUMBER[config["binder_chain"]]
    best_sequence = extract_sequence_from_batch(best_batch, binder_chain_num, alphabet)
    print("Best sequence:", best_sequence)
    
    # Save YAML configs
    result_yaml, result_yaml_apo = save_yaml_configs(
        yaml_path, best_sequence, binder_chain_num,
        directories['results_yaml'], directories['results_yaml_apo'],
        target_name, itr, config['length']
    )
    
    # Run predictions or save confidence scores
    if redo_boltz_predict:
        run_boltz_predictions(
            boltz_path, boltz_model_version, result_yaml, result_yaml_apo,
            directories['results_final'], directories['results_final_apo']
        )
    else:
        save_confidence_scores(
            directories['results_final'], output, best_structure,
            f"{target_name}_results_itr{itr + 1}_length{config['length']}",
            0, boltz2=(boltz_model_version == "boltz2")
        )
        save_confidence_scores(
            directories['results_final_apo'], output_apo, best_structure_apo,
            f"{target_name}_results_itr{itr + 1}_length{config['length']}",
            0, boltz2=(boltz_model_version == "boltz2")
        )
    
    return rmsd