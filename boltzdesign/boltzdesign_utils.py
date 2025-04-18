import os
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

import subprocess
import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
import copy
import click
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Connection, Input, Manifest, Record, Structure
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
from boltz.main import BoltzProcessedInput, BoltzDiffusionParams
from boltz.data.feature import featurizer
from boltz.data.tokenize.boltz import BoltzTokenizer, TokenData
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.parse.schema import parse_boltz_schema

from rdkit import Chem
import rdkit
import yaml
import shutil
from Bio import PDB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
import csv

def visualize_training_history(loss_history, sequence_history, distogram_history, length, save_dir=None, save_filename=None, binder_chain='A'):
    """
    Visualize training history including loss plot, distogram animation, and sequence evolution animation.
    
    Args:
        loss_history (list): List of loss values over training
        sequence_history (list): List of sequence probability matrices over training
        distogram_history (list): List of distogram matrices over training
        length (int): Length of sequence to visualize
        save_dir (str): Directory to save visualizations
    """
    # Trim sequence history to specified length
    if binder_chain == 'A':
        sequence_history = [seq[:length,:] for seq in sequence_history]
    elif binder_chain == 'B':
        sequence_history = [seq[-length:,:] for seq in sequence_history]

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Plot loss history
    plt.figure(figsize=(8,4))
    plt.plot(loss_history)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{save_filename}_loss_history.png'))
    plt.show()

    # Create distogram animation
    def create_distogram_animation():
        fig, ax = plt.subplots(figsize=(6,6))
        distogram_2d = np.argmax(distogram_history[0], axis=-1)
        im = ax.imshow(distogram_2d, cmap='viridis')
        plt.colorbar(im, ax=ax)
        ax.set_title('Distogram Evolution (Argmax)')

        def update(frame):
            distogram_2d = np.argmax(distogram_history[frame], axis=-1)
            im.set_data(distogram_2d)
            ax.set_title(f'Distogram Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(distogram_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_distogram_evolution.mp4'), writer='ffmpeg', fps=5)
        plt.close()
        return ani

    # Create sequence evolution animation
    def create_sequence_animation():
        fig, ax = plt.subplots(figsize=(12,3.5))
        im = ax.imshow(sequence_history[0].T, vmin=0, vmax=1, cmap='Blues', aspect='auto', alpha=0.8)
        plt.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(20))
        ax.set_yticklabels(list('ARNDCQEGHILKMFPSTWYV'))
        ax.set_title('Sequence Evolution')

        def update(frame):
            im.set_data(sequence_history[frame].T)
            ax.set_title(f'Sequence Epoch {frame + 1}')
            return im,

        ani = FuncAnimation(fig, update, frames=len(sequence_history), interval=200)
        if save_dir:
            ani.save(os.path.join(save_dir, f'{save_filename}_sequence_evolution.mp4'), writer='ffmpeg', fps=5)
        plt.close()
        return ani

    # Create and save animations
    distogram_ani = create_distogram_animation()
    sequence_ani = create_sequence_animation()



tokens = [
    "<pad>",
    "-",
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]


def get_mid_points(pdistogram):
    boundaries = torch.linspace(2, 22.0, 63)
    lower = torch.tensor([1.0])
    upper = torch.tensor([22.0 + 5.0])
    exp_boundaries = torch.cat((lower, boundaries, upper))
    mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
        pdistogram.device
    )

    return mid_points

import numpy as np

def np_kabsch(a, b, return_v=False):
    '''Get alignment matrix for two sets of coordinates using numpy
    3929
    
    Args:
        a: First set of coordinates
        b: Second set of coordinates
        return_v: If True, return U matrix from SVD. If False, return rotation matrix
        
    Returns:
        Rotation matrix (or U matrix if return_v=True) to align coordinates
    '''
    # Calculate covariance matrix
    ab = np.swapaxes(a, -1, -2) @ b
    
    # Singular value decomposition
    u, s, vh = np.linalg.svd(ab, full_matrices=False)
    
    # Handle reflection case
    flip = np.linalg.det(u @ vh) < 0
    if flip:
        u[...,-1] = -u[...,-1]
    
    return u if return_v else (u @ vh)

def np_rmsd(true, pred):
    '''Compute RMSD of coordinates after alignment using numpy
    
    Args:
        true: Reference coordinates
        pred: Predicted coordinates to align
        
    Returns:
        Root mean square deviation after optimal alignment
    '''
    # Center coordinates
    p = true - np.mean(true, axis=-2, keepdims=True)
    q = pred - np.mean(pred, axis=-2, keepdims=True)
    
    # Get optimal rotation matrix and apply it
    p = p @ np_kabsch(p, q)
    
    # Calculate RMSD
    return np.sqrt(np.mean(np.sum(np.square(p-q), axis=-1)) + 1e-8)

    
def min_k(x, k=1, mask=None):
    # Convert mask to boolean if it's not None
    if mask is not None:
        mask = mask.bool()  # Convert to boolean tensor
    
    # Sort the tensor, replacing masked values with Nan
    y = torch.sort(x if mask is None else torch.where(mask, x, float('nan')))[0]

    # Create a mask for the top k value
    k_mask = (torch.arange(y.shape[-1]).to(y.device) < k) & (~torch.isnan(y))
    # Compute the mean of the top k values
    return torch.where(k_mask, y, 0).sum(-1) / (k_mask.sum(-1) + 1e-8)

def get_con_loss(dgram, dgram_bins, num=None, seqsep=None, num_pos = float("inf"), cutoff=None, binary=False, mask_1d=None, mask_1b=None):
    con_loss = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    idx = torch.arange(dgram.shape[1])
    offset = idx[:,None] - idx[None,:]
    # Add mask for position separation > 3

    m =(torch.abs(offset)>=seqsep).to(dgram.device)
    if mask_1d is None: mask_1d = torch.ones(m.shape[0])
    if mask_1b is None: mask_1b = torch.ones(m.shape[0])

    m = torch.logical_and(m, mask_1b)
    p = min_k(con_loss, num, m).to(dgram.device)
    p = min_k(p, num_pos, mask_1d).to(dgram.device)
    return p

def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=False):
    '''dgram to contacts'''
    if cutoff is None: cutoff = dgram_bins[-1]
    bins = dgram_bins < cutoff  
    px = torch.softmax(dgram, dim=-1)
    px_ = torch.softmax(dgram - 1e7 * (~ bins), dim=-1)        
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

import matplotlib.pyplot as plt
def get_pae_loss(pae, mask_1d=None, mask_1b=None, mask_2d=None):
  pae = pae/31.0
  L = pae.shape[1]
  if mask_1d is None: mask_1d = torch.ones(L).to(pae.device)
  if mask_1b is None: mask_1b = torch.ones(L).to(pae.device)
  if mask_2d is None: mask_2d = torch.ones((L, L)).to(pae.device)
  mask_2d = mask_2d * mask_1d[:, :, None] * mask_1b[:, None, :]
  return mask_loss(pae, mask_2d)



def _get_helix_loss(dgram, dgram_bins, offset=None, mask_2d=None, binary=False, **kwargs):
    '''helix bias loss'''
    x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=binary)
    if offset is None:
        if mask_2d is None:
            return x.diagonal(offset=3).mean()
        else:
            mask_2d = mask_2d.float() 
            return (x * mask_2d).diagonal(offset=3, dim1=-2, dim2=-1).sum() / (torch.diagonal(mask_2d, offset=3, dim1=-2, dim2=-1).sum() + 1e-8)

    else:
        mask = (offset == 3).float()
        if mask_2d is not None:
            mask = mask * mask_2d.float()
        return (x * mask).sum() / (mask.sum() + 1e-8)
    


def add_rg_loss(sample_atom_coords, atom_to_token, length, binder_chain='A'):
    atom_order = torch.cumsum(atom_to_token, dim=1)
    ca_mask = torch.sum((atom_order == 2).to(atom_to_token.dtype), dim=-1)[0]
    if binder_chain == 'A':
        ca_coords = sample_atom_coords[:,ca_mask==1,:][:,:length,:]
    elif binder_chain == 'B':
        ca_coords = sample_atom_coords[:,ca_mask==1,:][:,-length:,:]
    center_of_mass = ca_coords.mean(1, keepdim=True)  # keepdim for proper broadcasting
    squared_distances = torch.sum(torch.square(ca_coords - center_of_mass), dim=-1)
    rg = torch.sqrt(squared_distances.mean() + 1e-8)
    rg_th = 2.38 * ca_coords.shape[1] ** 0.365
    loss = torch.nn.functional.elu(rg - rg_th)
    
    return loss




def get_boltz_model(checkpoint: Optional[str] = None, predict_args=None, device: Optional[str] = None) -> Boltz1:
    torch.set_grad_enabled(True)
    torch.set_float32_matmul_precision("highest")
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value
    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=False,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        structure_prediction_training=True,
        no_msa=False,
        no_atom_encoder=False, 
    )
    return model_module


def get_coords(batch, output):
    token_to_rep = batch['token_to_rep_atom'][0].detach().cpu()  # Fixed missing parentheses
    indices = [torch.argmax(token_to_rep[i]).item() for i in range(token_to_rep.shape[0])]
    sample_coords = output['sample_atom_coords'].detach()
    sample_coords = sample_coords.cpu()
    coords = sample_coords.numpy()[0, indices, :]
    return coords





def boltz_hallucination_4stages_dist_only_clearner(
    # Required arguments
    boltz_model,
    yaml_path,
    ccd_lib,
    length=100,
    binder_chain='A',
    design_algorithm="3stages",
    pre_iteration=20,
    soft_iteration=50, 
    temp_iteration=50,
    hard_iteration=10,
    semi_greedy_steps=0,
    learning_rate=1.0,
    mutation_rate=1,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    use_temp=False,
    disconnect_feats=False,
    disconnect_pairformer=False,
    mask_ligand=False,
    distogram_only=False,
    input_res_type=False,
    small_molecule=False,
    loss_scales=None,
    optimize_per_contact_per_binder_pos=False
):

    predict_args = {
        "recycling_steps": 0,  # Default value
        "sampling_steps": 200,  # Default value
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    with yaml_path.open("r") as file:
        data = yaml.safe_load(file)

    if binder_chain == 'A':
        data['sequences'][0]['protein']['sequence'] = 'X'*length
    elif binder_chain == 'B':
        data['sequences'][1]['protein']['sequence'] = 'X'*length
        
    name = yaml_path.stem
    target = parse_boltz_schema(name, data, ccd_lib)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if set_train:
        boltz_model.train()
        print("set in train mode")
    else:
        boltz_model.eval()
        print("set in eval mode")

    def get_batch(target, max_seqs=0, length=100):
        target_id = target.record.id
        structure = target.structure

        structure = Structure(
                atoms=structure.atoms,
                bonds=structure.bonds,
                residues=structure.residues,
                chains=structure.chains,
                connections=structure.connections.astype(Connection),
                interfaces=structure.interfaces,
                mask=structure.mask,
            )

        msas = {}
        for chain in target.record.chains:
            msa_id = chain.msa_id
            # Load the MSA for this chain, if any
            if msa_id != -1:
                msa = np.load(msa_id)
                msas[chain.chain_id] = MSA(**msa)

        input = Input(structure, msas) ## currently MSA to be None
        
        tokenizer = BoltzTokenizer()
        tokenized = tokenizer.tokenize(input)
        featurizer = BoltzFeaturizer()

        batch = featurizer.process(
                    tokenized,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=max_seqs,
                    pad_to_max_seqs=False,
                    symmetries={},
                    compute_symmetries=False,
                    inference_binder=None,
                    inference_pocket=None,
                )
        return batch
    
    # with torch.autograd.profiler.profile(use_cuda=True) as prof:

    batch = get_batch(target, max_seqs=4096, length=length)
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}

    ## initialize res_type_logits
    if not pre_run:
        batch['res_type_logits'] = torch.from_numpy(input_res_type).to(device)
        print("Initialize with pre-run input_res_type", input_res_type)
    else:
        batch['res_type_logits'] = batch['res_type'].clone().detach().to(device).float()
        if binder_chain == 'A':
            batch['res_type_logits'][:, :length,:] = 0.1*torch.distributions.Gumbel(0, 1).sample(batch['res_type'][:, :length,:].shape).to(device)
        elif binder_chain == 'B':
            batch['res_type_logits'][:, -length:,:] = 0.1*torch.distributions.Gumbel(0, 1).sample(batch['res_type'][:, -length:,:].shape).to(device)
        print("Pre run- input_res_type", batch['res_type_logits'])

    batch['msa'] = batch['res_type_logits'].unsqueeze(0).to(device)
    batch['msa_paired'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
    batch['deletion_value'] = torch.zeros(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
    batch['has_deletion'] = torch.full((batch['res_type'].shape[0], 1, batch['res_type'].shape[1]), False).to(device)  
    batch['msa_mask'] = torch.ones(batch['res_type'].shape[0], 1, batch['res_type'].shape[1]).to(device)
    batch['profile'] = batch['msa'].float().mean(dim=0).to(device)
    batch['deletion_mean'] = torch.zeros(batch['deletion_mean'].shape).to(device)
    batch['res_type'] = batch['res_type'].float()  # Ensure the tensor is of floating point type
    batch['res_type_logits'].requires_grad = True
    optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
    num_pos = batch['res_type'].shape[1] - length

    def norm_seq_grad(grad, chain_mask):
        chain_mask = chain_mask.bool()
        masked_grad = grad[:, chain_mask.squeeze(0), :] 
        eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
        gn = masked_grad.norm(dim=(-1, -2), keepdim=True) 
        return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    best_loss = float('inf')  # Track the best loss
    min_loss = float('inf')  # Track the minimum loss observed\
    best_batch = None  # Variable to save the best batch
    first_step_best_batch=None

    plots = []
    plots_from_structure_module=[]
    plots_pae=[]
    plots_res_type=[]

    distogram_history = []
    sequence_history = []
    loss_history = []
    lr_history = []
    con_loss_history = []
    i_con_loss_history = []
    plddt_loss_history = []

    if binder_chain == 'A':
        mask = torch.ones_like(batch['res_type_logits'])
        mask[:, length:, :] = 0
    elif binder_chain == 'B':
        mask = torch.ones_like(batch['res_type_logits'])
        mask[:, :-length, :] = 0
        
    
    prev_sequence = None
    mid_points = torch.linspace(2, 22, 64).to(device) 
    chain_mask = (batch['entity_id'] == 0).int()

    def design(batch, 
               iters = None,
                soft=0.0, e_soft=None,
                step=1.0, e_step=None,
                temp=1.0, e_temp=None,
                hard=0.0, e_hard=None,
                num_contacts=1, e_num_contacts=1,
                learning_rate=1.0,
                mask=None,
                chain_mask=None,
                length=100,
                plots=None,
                loss_history=None,
                i_con_loss_history=None,
                con_loss_history=None,
                plddt_loss_history=None,
                distogram_history=None,
                sequence_history=None,
                pre_run=False,
                mask_ligand=False,
                distogram_only=False,
                predict_args=None,
                alpha=2.0,
                loss_scales=None,
                binder_chain='A',
                small_molecule=False,
                optimize_per_contact_per_binder_pos=False,
                ):

        prev_sequence=""
        def get_model_loss(batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, pre_run=False, mask_ligand=False, distogram_only=False, predict_args=None, loss_scales=None, binder_chain='A', small_molecule=False, optimize_per_contact_per_binder_pos=False, num_contacts=1):
            # Get model outputs based on pre_run flag
            if pre_run:
                if binder_chain == 'A':
                    if mask_ligand:
                        batch['token_pad_mask'][:,length:]=0
                        batch['atom_pad_mask'][:,5*length:]=0
                elif binder_chain == 'B':
                    if mask_ligand:
                        batch['token_pad_mask'][:,:-length]=0
                        batch['atom_pad_mask'][:,:-5*length]=0
                dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
            else:
                if distogram_only:
                    dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
                else:
                    dict_out = boltz_model.design_pairformer_confidence(batch, 
                        recycling_steps=predict_args["recycling_steps"],
                        num_sampling_steps=predict_args["sampling_steps"], 
                        multiplicity_diffusion_train=1,
                        diffusion_samples=predict_args["diffusion_samples"],
                        run_confidence_sequentially=True,
                        disconnect_feats=disconnect_feats,
                        disconnect_pairformer=disconnect_pairformer)

            # Get predicted distance distribution and midpoints
            pdist = dict_out['pdistogram']
            mid_pts = get_mid_points(pdist).to(device)

            # Calculate contact losses
            con_loss = get_con_loss(pdist, mid_pts,
                                num=3, seqsep=9, cutoff=14.0,
                                binary=False,
                                mask_1d=chain_mask, mask_1b=chain_mask)

        
            if small_molecule:
                if optimize_per_contact_per_binder_pos:
                    num_contacts = 0 if pre_run else num_contacts
                    print(f"num_contacts: {num_contacts}")
                    i_con_loss = get_con_loss(pdist, mid_pts,
                                            num=2, seqsep=0, num_pos=num_contacts,
                                            cutoff=14.0, binary=False, 
                                            mask_1d=chain_mask, mask_1b=1-chain_mask)
                else:
                    num_contacts = 0 if pre_run else float('inf')
                    i_con_loss = get_con_loss(pdist, mid_pts,
                                            num=2, seqsep=0, num_pos=num_contacts,
                                            cutoff=14.0, binary=False, 
                                            mask_1d=1-chain_mask, mask_1b=chain_mask)
            else:
                i_con_loss = get_con_loss(pdist, mid_pts,
                                        num=1, seqsep=0,
                                        cutoff=21.0, binary=False, 
                                        mask_1d=chain_mask, mask_1b=1-chain_mask)

            # Calculate helix loss with 2D mask
            mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
            helix_loss = _get_helix_loss(pdist, mid_pts,
                                    offset=None, mask_2d=mask_2d, binary=True)

            # Initialize losses dict with common losses
            losses = {
                'con_loss': con_loss,
                'i_con_loss': i_con_loss,
                'helix_loss': helix_loss
            }

            # Add confidence-based losses if not in pre-run
            if not pre_run:
                if not distogram_only:
                    plddt_loss = get_plddt_loss(dict_out['plddt'], mask_1d=chain_mask)
                    pae = (dict_out['pae'] + dict_out['pae'].transpose(-2,-1))/2
                    i_pae_loss = get_pae_loss(pae, mask_1d=1-chain_mask, mask_1b=chain_mask)
                    pae_loss = get_pae_loss(pae, mask_1d=chain_mask, mask_1b=chain_mask)
                    rg_loss = add_rg_loss(dict_out['sample_atom_coords'], batch['atom_to_token'], length, binder_chain=binder_chain)
                    
                    losses.update({
                        'plddt_loss': plddt_loss,
                        'i_pae_loss': i_pae_loss,
                        'pae_loss': pae_loss,
                        'rg_loss': rg_loss
                    })
                    
                    plddt_loss_history.append(plddt_loss.item())

            # Calculate contact probability
            bins = mid_points < 8.0
            px = torch.sum(torch.softmax(dict_out['pdistogram'], dim=-1)[:,:,:,bins], dim=-1)

            if loss_scales is None:
                loss_scales = {
                    'con_loss': 1.0,
                    'i_con_loss': 1.0, 
                    'helix_loss': 0.0,
                    'plddt_loss': 0.1,
                    'pae_loss': 0.4,
                    'i_pae_loss': 0.1,
                    'rg_loss': 0.4,
                }

            # Calculate total loss and print individual losses
            total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
            print([f"{k}:{v.item():.2f}" for k,v in losses.items()])

            # Update histories
            plots.append(px[0].detach().cpu().numpy())
            loss_history.append(total_loss.item())
            i_con_loss_history.append(i_con_loss.item())
            con_loss_history.append(con_loss.item())
            distogram_history.append(torch.softmax(dict_out['pdistogram'], dim=-1)[0].detach().cpu().numpy())
            sequence_history.append(batch['res_type'][0, :, 2:22].detach().cpu().numpy())

            return total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history
        
        def update_sequence(opt, batch, mask, alpha=2.0):
            batch["logits"] = alpha*batch['res_type_logits']
            X =  batch['logits']- torch.sum(torch.eye(batch['logits'].shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0).to(device)*(1e10)
            batch['soft'] = torch.softmax(X/opt["temp"],dim=-1)
            batch['hard'] =  torch.zeros_like(batch['soft']).scatter_(-1, batch['soft'].max(dim=-1, keepdim=True)[1], 1.0)
            batch['hard'] =  (batch['hard'] - batch['soft']).detach() + batch['soft']
            batch['pseudo'] =  opt["soft"] * batch["soft"] + (1-opt["soft"]) * batch["res_type_logits"]
            batch['pseudo'] = opt["hard"] * batch["hard"] + (1-opt["hard"]) * batch["pseudo"]
            batch['res_type'] = batch['pseudo']*mask + batch['res_type_logits']*(1-mask)
            batch['msa'] = batch['res_type'].unsqueeze(0).to(device).detach()
            batch['profile'] = batch['msa'].float().mean(dim=0).to(device).detach()
            return batch
        
        m = {"soft":[soft,e_soft],"temp":[temp,e_temp],"hard":[hard,e_hard], "step":[step,e_step], 'num_contacts':[num_contacts, e_num_contacts]}
        m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

        opt = {}
        for i in range(iters):
            for k,(s,e) in m.items():
                if k == "temp":
                    opt[k] = (e+(s-e)*(1-(i)/iters)**2)
                else:
                    v = (s+(e-s)*((i)/iters))
                    if k == "step": step = v
                    opt[k] = v
                
            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_contacts = int(opt["num_contacts"])

            for param_group in optimizer.param_groups:
                print(f"learning_rate: {learning_rate:.2f}, lr_scale: {lr_scale:.2f}")
                param_group['lr'] = learning_rate * lr_scale

            opt["lr_rate"] = learning_rate * lr_scale
                
            batch = update_sequence(opt, batch, mask)

            total_loss, plots, loss_history, i_con_loss_history, con_loss_history, distogram_history, sequence_history, plddt_loss_history = get_model_loss(batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history, pre_run, mask_ligand, distogram_only, predict_args, loss_scales, binder_chain, small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos, num_contacts=num_contacts)
            if binder_chain == 'A':
                current_sequence = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][:, :length, :], dim=-1)[0].detach().cpu().numpy()])
            elif binder_chain == 'B':
                current_sequence = ''.join([alphabet[i] for i in torch.argmax(batch['res_type'][:, -length:, :], dim=-1)[0].detach().cpu().numpy()])
            if prev_sequence is not None:
                diff_count = sum(1 for a, b in zip(current_sequence, prev_sequence) if a != b)
                diff_percentage = (diff_count / length) * 100
                print(f"Sequence difference from previous: {diff_percentage:.2f}%")
            prev_sequence = current_sequence

            total_loss.backward()
            if batch['res_type_logits'].grad is not None:
                if binder_chain == 'A':
                    batch['res_type_logits'].grad[:,length:,:] = 0
                    batch['res_type_logits'].grad[:, :length, [0,1,6,22,23,24,25,26,27,28,29,30,31,32]] = 0
                elif binder_chain == 'B':
                    batch['res_type_logits'].grad[:, :-length,:] = 0
                    batch['res_type_logits'].grad[:, -length:, [0,1,6,22,23,24,25,26,27,28,29,30,31,32]] = 0
                batch['res_type_logits'].grad = norm_seq_grad(batch['res_type_logits'].grad, chain_mask)
                optimizer.step()
                optimizer.zero_grad()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {i}: Loss: {total_loss.item():.2f}, Current learning rate: {current_lr:.2f}")


        return batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history

    if pre_run:
        batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history = design(batch, iters=pre_iteration, soft=1.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, mask_ligand=mask_ligand, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, small_molecule=small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos)
    else:
        if design_algorithm == "3stages":
            print('-'*100)
            print("logits to softmax(T=1)")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history = design(batch, iters=soft_iteration, e_soft=0.8, num_contacts=1, e_num_contacts=8, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, small_molecule=small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos)

            print('-'*100)
            print("softmax(T=1) to softmax(T=0.01)")
            print('-'*100)
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch['res_type_logits'] = new_logits
            optimizer = torch.optim.SGD([batch['res_type_logits']], lr=learning_rate)
            batch, plots, loss_history, i_con_loss_history, con_loss_history,plddt_loss_history, distogram_history, sequence_history = design(batch, iters=temp_iteration, soft=1.0, temp = 1.0,e_temp=0.01, num_contacts=8, e_num_contacts=12,  mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, small_molecule=small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos)
            print('-'*100)
            print("hard")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history = design(batch, iters=hard_iteration, soft=1.0, hard = 1.0,temp=0.01, num_contacts=12, e_num_contacts=16, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, small_molecule=small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos)

        elif design_algorithm == "logits":
            print('-'*100)
            print("logits")
            print('-'*100)
            batch, plots, loss_history, i_con_loss_history, con_loss_history, plddt_loss_history, distogram_history, sequence_history = design(batch, iters=soft_iteration, soft = 0.0, e_soft=0.0, mask=mask, chain_mask=chain_mask, learning_rate=learning_rate, length=length, plots=plots, loss_history=loss_history, i_con_loss_history=i_con_loss_history, con_loss_history=con_loss_history, plddt_loss_history=plddt_loss_history, distogram_history=distogram_history, sequence_history=sequence_history, pre_run=pre_run, distogram_only=distogram_only, predict_args=predict_args, loss_scales=loss_scales, binder_chain=binder_chain, small_molecule=small_molecule, optimize_per_contact_per_binder_pos=optimize_per_contact_per_binder_pos)


    def visualize_results(plots, plots_res_type, plots_from_structure_module, plots_pae):
        # Plot distogram predictions
        if plots:
            num_plots = len(plots)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
            
            if num_rows == 1:
                axs = axs.reshape(1, -1)

            for i, plot_data in enumerate(plots):
                row, col = i // 6, i % 6
                axs[row, col].imshow(plot_data)
                axs[row, col].set_title(f'Epoch {i + 1}')
                axs[row, col].axis('off')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            
            plt.show()
            plots.clear()

        # Plot res_type histograms
        if plots_res_type:
            num_plots = len(plots_res_type)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
            
            if num_rows == 1:
                axs = axs.reshape(1, -1)
                
            for i, plot_data in enumerate(plots_res_type):
                row, col = i // 6, i % 6
                axs[row, col].imshow(plot_data.T)
                axs[row, col].set_title(f'Res Type Epoch {i + 1}')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            plt.show()
            plots_res_type.clear()

        # Plot structure module predictions
        if plots_from_structure_module:
            num_plots = len(plots_from_structure_module)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))

            for i, plot_data in enumerate(plots_from_structure_module):
                ax_index = i if num_rows == 1 else i // 6
                ax = axs[ax_index] if num_rows == 1 else axs[ax_index, i % 6]
                ax.imshow(plot_data)
                ax.set_title(f'Structure Module Epoch {i + 1}')
                ax.axis('off')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                if num_rows == 1:
                    axs[j].axis('off')
                else:
                    axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            plt.show()
            plots_from_structure_module.clear()

        # Plot PAE
        if plots_pae:
            num_plots = len(plots_pae)
            num_rows = (num_plots + 5) // 6
            fig, axs = plt.subplots(num_rows, 6, figsize=(15, num_rows * 2.5))
            for i, plot_data in enumerate(plots_pae):
                ax_index = i if num_rows == 1 else i // 6
                ax = axs[ax_index] if num_rows == 1 else axs[ax_index, i % 6]
                ax.imshow(plot_data, cmap='bwr')
                ax.set_title(f'PAE Epoch {i + 1}')
                ax.axis('off')

            # Hide unused subplots
            for j in range(num_plots, num_rows * 6):
                if num_rows == 1:
                    axs[j].axis('off')
                else:
                    axs[j // 6, j % 6].axis('off')

            plt.tight_layout()
            plt.show()
            plots_pae.clear()

    # Call visualization function if needed
    visualize_results(plots, plots_res_type, plots_from_structure_module, plots_pae)

    if pre_run:
        return batch['res_type'].detach().cpu().numpy(), plots, loss_history, distogram_history, sequence_history 

    print("Designing complete")
    boltz_model.eval()

    if best_batch is None:
        if first_step_best_batch is not None:
            best_batch = first_step_best_batch
        else:
            best_batch = batch  

    predict_args = {
    "recycling_steps": 3,  # Default value
    "sampling_steps": 200,  # Default value
    "diffusion_samples": 1,  # Default value
    "write_confidence_summary": True,
    "write_full_pae": False,
    "write_full_pde": False,
    }

    def _mutate(sequence, best_logits, i_prob):
        mutated_sequence = list(sequence) # Create a copy of the input tensor
        for m in range(1):
            i = np.random.choice(np.arange(length),p=i_prob/i_prob.sum())
            i_logits = best_logits[:, i]
            i_logits = i_logits - torch.max(i_logits)
            i_X = i_logits- (torch.sum(torch.eye(i_logits.shape[-1])[[0,1,6,22,23,24,25,26,27,28,29,30,31,32]],dim=0)*(1e10)).to(device)
            i_aa = torch.multinomial(torch.softmax(i_X, dim=-1), 1).item()
            print("position", i, "before mutation", mutated_sequence[i], "after mutation", alphabet[i_aa])
            mutated_sequence[i] = alphabet[i_aa]
        return ''.join(mutated_sequence)

    best_logits = best_batch['res_type_logits']
    if binder_chain == 'A':
        best_seq= ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][:, :length, :], dim=-1)[0].detach().cpu().numpy()])
    elif binder_chain == 'B':
        best_seq= ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][:, -length:, :], dim=-1)[0].detach().cpu().numpy()])
    
    if binder_chain == 'A':
        data['sequences'][0]['protein']['sequence'] = best_seq
    elif binder_chain == 'B':
        data['sequences'][1]['protein']['sequence'] = best_seq

    data_apo = copy.deepcopy(data)  # This handles all types of values correctly
    del data_apo['sequences'][1]

    target = parse_boltz_schema(name, data, ccd_lib)
    target_apo = parse_boltz_schema(name, data_apo, ccd_lib)
    best_batch = get_batch(target, 4096, length)
    best_batch_apo = get_batch(target_apo, 4096, length)
    best_batch = {key: value.unsqueeze(0).to(device) for key, value in best_batch.items()}
    best_batch_apo = {key: value.unsqueeze(0).to(device) for key, value in best_batch_apo.items()}

    output = boltz_model( 
            best_batch,
            recycling_steps = predict_args["recycling_steps"],
            num_sampling_steps=predict_args["sampling_steps"],
            multiplicity_diffusion_train=1,
            diffusion_samples=predict_args["diffusion_samples"],
            run_confidence_sequentially=True)

   
    output_apo = boltz_model( 
            best_batch_apo,
            recycling_steps = predict_args["recycling_steps"],
            num_sampling_steps=predict_args["sampling_steps"],
            multiplicity_diffusion_train=1,
            diffusion_samples=predict_args["diffusion_samples"],
            run_confidence_sequentially=True)

    for _ in range(semi_greedy_steps):
        plddt_loss = []
        confidence_score=[]
        mutated_sequence_ls=[]
        for t in range(10): 
            if binder_chain == 'A':
                plddt = output['plddt'][0][:length]
                i_prob = np.ones(length) if plddt is None else torch.maximum(1-plddt,torch.tensor(0))
                i_prob = i_prob.detach().cpu().numpy() if torch.is_tensor(i_prob) else i_prob
                sequence = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][:, :length, :], dim=-1)[0].detach().cpu().numpy()])
            elif binder_chain == 'B':
                plddt = output['plddt'][0][-length:]
                i_prob = np.ones(length) if plddt is None else torch.maximum(1-plddt,torch.tensor(0))
                i_prob = i_prob.detach().cpu().numpy() if torch.is_tensor(i_prob) else i_prob
                sequence = ''.join([alphabet[i] for i in torch.argmax(best_batch['res_type'][:, -length:, :], dim=-1)[0].detach().cpu().numpy()])
            
            print("sequence", sequence)
            mutated_sequence  = _mutate(sequence, best_logits, i_prob)
            print("mutated_sequence", mutated_sequence)
            data['sequences'][0]['protein']['sequence'] = mutated_sequence
            target = parse_boltz_schema(name, data, ccd_lib)
            best_batch = get_batch(target, 4096, length)
            best_batch = {key: value.unsqueeze(0).to(device) for key, value in best_batch.items()}
            output = boltz_model( 
                    best_batch,
                    recycling_steps = predict_args["recycling_steps"],
                    num_sampling_steps=predict_args["sampling_steps"],
                    multiplicity_diffusion_train=1,
                    diffusion_samples=predict_args["diffusion_samples"],
                    run_confidence_sequentially=True)
                    
            confidence  =output['complex_plddt'].detach().cpu().numpy()*0.8 + output['ligand_iptm'].detach().cpu().numpy()*0.2
            if binder_chain == 'A':
                plddt_loss.append(output['plddt'][0].detach().cpu().numpy()[:length])
            elif binder_chain == 'B':
                plddt_loss.append(output['plddt'][0].detach().cpu().numpy()[-length:])
            confidence_score.append(confidence)
            mutated_sequence_ls.append(mutated_sequence)

        best_id = np.argmax(confidence_score)
        best_seq= mutated_sequence_ls[best_id]
        data['sequences'][0]['protein']['sequence'] = best_seq
        data_apo['sequences'][0]['protein']['sequence'] = best_seq
        target = parse_boltz_schema(name, data, ccd_lib)
        target_apo = parse_boltz_schema(name, data_apo, ccd_lib)
        best_batch = get_batch(target, 4096, length)
        best_batch_apo = get_batch(target_apo, 4096, length)
        best_batch = {key: value.unsqueeze(0).to(device) for key, value in best_batch.items()}
        best_batch_apo = {key: value.unsqueeze(0).to(device) for key, value in best_batch_apo.items()}

    return output, output_apo, best_batch, best_batch_apo, distogram_history, sequence_history, loss_history, con_loss_history, i_con_loss_history, plddt_loss_history

def run_boltz_design(
    main_dir,
    yaml_dir,
    boltz_model,
    ccd_lib,
    design_samples =1,
    version_name=None,
    config=None,
    loss_scales=None,
):
    """
    Run Boltz protein design pipeline.
    
    Args:
        main_dir (str): Main directory path
        yaml_dir (str): Directory containing input yaml files
        version_name (str): Name for version subdirectory
        config (dict): Configuration parameters. If None, uses defaults.
    """
    if config is None:
        config = {
            'mutation_rate': 1,
            'pre_iteration': 30,
            'soft_iteration': 75, 
            'temp_iteration': 45,
            'hard_iteration': 5,
            'semi_greedy_steps': 0,
            'learning_rate_pre': 0.2,
            'learning_rate': 0.1,
            'design_algorithm': '3stages',
            'set_train': True,
            'use_temp': True,
            'disconnect_feats': True,
            'disconnect_pairformer': False,
            'distogram_only': True,
            'length': 150,
            'binder_chain': 'A',
            'small_molecule': False,
            'mask_ligand': False,
            'optimize_per_contact_per_binder_pos':False
        }

    version_dir = os.path.join(main_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    results_final_dir = os.path.join(version_dir, 'results_final')
    results_yaml_dir = os.path.join(version_dir, 'results_yaml')
    apo_dir = os.path.join(version_dir, 'results_apo_final')
    apo_yaml_dir = os.path.join(version_dir, 'results_apo_yaml')
    loss_dir = os.path.join(version_dir, 'loss')
    animation_save_dir = os.path.join(version_dir, 'animation')

    for directory in [results_yaml_dir, results_final_dir, apo_yaml_dir, 
                     apo_dir, loss_dir, animation_save_dir]:
        os.makedirs(directory, exist_ok=True)

    # Save config
    config_path = os.path.join(results_final_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    alphabet = list('XXARNDCQEGHILKMFPSTWYV-')
    rmsd_csv_path = os.path.join(results_final_dir, 'rmsd_results.csv')
    csv_exists = os.path.exists(rmsd_csv_path)

    for yaml_path in Path(yaml_dir).glob('*.yaml'):
        if yaml_path.name.endswith('.yaml'):
            try:
                target_binder_input = yaml_path.stem
                for itr in range(design_samples):
                    torch.cuda.empty_cache()
                    output = None
                    while output is None:  # Keep trying until we get valid output
                        print('pre-run warm up')
                        input_res_type, plots, loss_history, distogram_history, sequence_history = boltz_hallucination_4stages_dist_only_clearner(
                            boltz_model,
                            yaml_path,
                            ccd_lib,
                            length=config['length'],
                            mutation_rate=config['mutation_rate'],
                            pre_run=True,
                            pre_iteration=config['pre_iteration'],
                            soft_iteration=config['soft_iteration'],
                            temp_iteration=config['temp_iteration'],
                            hard_iteration=config['hard_iteration'],
                            semi_greedy_steps=config['semi_greedy_steps'],
                            learning_rate=config['learning_rate_pre'],
                            disconnect_feats=config['disconnect_feats'],
                            disconnect_pairformer=config['disconnect_pairformer'],
                            set_train=config['set_train'],
                            use_temp=config['use_temp'],
                            distogram_only=True,
                            input_res_type=False,
                            loss_scales=loss_scales,
                            binder_chain=config['binder_chain'],
                            small_molecule=config['small_molecule'],
                            mask_ligand=config['mask_ligand'],
                            optimize_per_contact_per_binder_pos=config['optimize_per_contact_per_binder_pos']
                        )
                        
                        print('hallucination')     

                        output, output_apo, best_batch, best_batch_apo, distogram_history_2, sequence_history_2, loss_history_2, con_loss_history, i_con_loss_history, plddt_loss_history = boltz_hallucination_4stages_dist_only_clearner(
                            boltz_model,
                            yaml_path,
                            ccd_lib,
                            length=config['length'],
                            mutation_rate=config['mutation_rate'],
                            pre_run=False,
                            design_algorithm=config['design_algorithm'],
                            pre_iteration=config['pre_iteration'],    
                            soft_iteration=config['soft_iteration'],
                            temp_iteration=config['temp_iteration'],
                            hard_iteration=config['hard_iteration'],
                            semi_greedy_steps=config['semi_greedy_steps'],
                            learning_rate=config['learning_rate'],
                            disconnect_feats=config['disconnect_feats'],
                            disconnect_pairformer=config['disconnect_pairformer'],
                            set_train=config['set_train'],
                            use_temp=config['use_temp'],
                            distogram_only= config['distogram_only'],
                            input_res_type=input_res_type,
                            loss_scales=loss_scales,
                            binder_chain=config['binder_chain'],
                            small_molecule=config['small_molecule'],
                            mask_ligand=config['mask_ligand'],
                            optimize_per_contact_per_binder_pos=config['optimize_per_contact_per_binder_pos']
                        )

                        loss_history.extend(loss_history_2)
                        distogram_history.extend(distogram_history_2) 
                        sequence_history.extend(sequence_history_2)
            
                        print('-' * 100)
                        print(f"Holo Protein PLDDT: {output['plddt'][:config['length']].mean()}")
                        print(f"Apo Protein PLDDT: {output_apo['plddt'][:config['length']].mean()}")
                        print('-' * 100)
                        print(f"Holo Complex PLDDT: {output['complex_plddt']}")
                        print(f"Apo Complex PLDDT: {output_apo['complex_plddt']}")
                        print('-' * 100)

                        coords = get_coords(best_batch, output)
                        coords_apo = get_coords(best_batch_apo, output_apo)

                        if config['binder_chain'] == 'A':
                            protein_coords = coords[:config['length'], :]
                            protein_coords_apo = coords_apo[:config['length'], :]
                        elif config['binder_chain'] == 'B':
                            protein_coords = coords[-config['length']: , :]
                            protein_coords_apo = coords_apo[-config['length']: , :]
                        
                        rmsd = np_rmsd(protein_coords, protein_coords_apo)
                        print('-' * 100)
                        print("rmsd", rmsd)
                        print('-' * 100)

                        if loss_dir:
                            os.makedirs(loss_dir, exist_ok=True)
                        # Plot loss history
                        try:
                            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
                            ax1.plot(loss_history)
                            ax1.set_xlabel('Steps')
                            ax1.set_ylabel('Loss')
                            ax1.set_title('Training Loss History')
                            ax2.plot(con_loss_history)
                            ax2.set_xlabel('Steps')
                            ax2.set_ylabel('Con Loss')
                            ax2.set_title('Con Loss History')
                            ax3.plot(i_con_loss_history)
                            ax3.set_xlabel('Steps')
                            ax3.set_ylabel('iCon Loss')
                            ax3.set_title('iCon Loss History')
                            
                            if loss_dir:
                                plt.savefig(os.path.join(loss_dir, f'{target_binder_input}_loss_history_itr{itr + 1}_length{config["length"]}.png'))
                            plt.show()
                            # visualize_training_history(loss_history, sequence_history, distogram_history, config["length"], save_dir=animation_save_dir, save_filename=f"{target_binder_input}_itr{itr + 1}_length{config['length']}", binder_chain=config['binder_chain'])

                        except Exception as e:
                            print(f"Error plotting loss history: {str(e)}")
                            continue

                        with open(rmsd_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not csv_exists:
                                writer.writerow(['target', 'length', 'iteration', 'rmsd', 'complex_plddt', 'ligand_iptm'])
                                csv_exists = True
                            writer.writerow([target_binder_input, config['length'], itr + 1, rmsd, output['complex_plddt'].item(), output['ligand_iptm'].item()])

                        result_yaml = os.path.join(results_yaml_dir, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                        apo_yaml = os.path.join(apo_yaml_dir, f'{target_binder_input}_results_itr{itr + 1}_length{config["length"]}.yaml')
                        output_cpu = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in output.items()}
                        best_batch_cpu = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in best_batch.items()}
                        if config['binder_chain'] == 'A':
                            best_sequence = ''.join([alphabet[i] for i in torch.argmax(torch.tensor(best_batch_cpu['res_type'][:, :config['length'], :]), dim=-1)[0].detach().cpu().numpy()])
                        elif config['binder_chain'] == 'B':
                            best_sequence = ''.join([alphabet[i] for i in torch.argmax(torch.tensor(best_batch_cpu['res_type'][:, -config['length']: , :]), dim=-1)[0].detach().cpu().numpy()])
                        
                        print("best_sequence", best_sequence)
                        shutil.copy2(yaml_path, result_yaml)
                        with open(result_yaml, 'r') as f:
                            data = yaml.safe_load(f)
                        if config['binder_chain'] == 'A':
                            data['sequences'][0]['protein']['sequence'] = best_sequence
                        elif config['binder_chain'] == 'B':
                            data['sequences'][1]['protein']['sequence'] = best_sequence
                        with open(result_yaml, 'w') as f:
                            yaml.dump(data, f)

                        subprocess.run(['boltz', 'predict', str(result_yaml), '--out_dir', str(results_final_dir), '--write_full_pae'])                     
                        print(f"Completed processing {target_binder_input} iteration {itr + 1}")

                        shutil.copy2(yaml_path, apo_yaml)
                        with open(apo_yaml, 'r') as f:
                            data = yaml.safe_load(f)
                        if config['binder_chain'] == 'A':
                            data['sequences'][0]['protein']['sequence'] = best_sequence
                        elif config['binder_chain'] == 'B':
                            data['sequences'][1]['protein']['sequence'] = best_sequence
                            
                        if 'ligand' in data['sequences'][1]:
                            del data['sequences'][1]
                        
                        with open(apo_yaml, 'w') as f:
                            yaml.dump(data, f)
                        subprocess.run(['boltz', 'predict', str(apo_yaml), '--out_dir', str(apo_dir), '--write_full_pae'])

                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {target_binder_input} iteration {itr + 1}: {str(e)}")
                continue