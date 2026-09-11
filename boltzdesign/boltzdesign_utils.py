import copy
import logging
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import yaml
from Bio.PDB import MMCIFParser, PDBParser
from IPython.display import HTML, display
from boltz.data.feature.featurizerv2 import Boltz2Featurizer
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.mol import load_canonicals, load_molecules
from boltz.data.parse.schema import parse_boltz_schema
from boltz.data.tokenize.boltz2 import Boltz2Tokenizer
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import (
    MSA,
    Coords,
    Ensemble,
    Input,
    Interface,
    StructureV2,
    Structure,
)
from boltz.data.write.mmcif import to_mmcif

from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
)
from boltz.model.models.boltz2 import Boltz2
from boltz.model.models.boltz1 import Boltz1
from utils import (
    np_kabsch, 
    np_rmsd,
    get_CA_and_sequence,
    align_points,
    get_mid_points,
    min_k,
    get_con_loss,
    mask_loss,
    get_plddt_loss,
    get_pae_loss,
    get_helix_loss,
    get_ca_coords,
    add_rg_loss,
    aggressive_memory_cleanup,
    print_memory_usage,
    save_trajectory_pdbs,
    get_omit_mask,
    get_motif_mask,
    update_batch_msa,
    extract_sequence_from_batch,
    save_confidence_scores,
    plot_loss_history,
    setup_output_directories,
    save_yaml_configs,
    plot_loss_history,
    cleanup_iteration,
    CHAIN_TO_NUMBER,
    process_design_results,
    shift_motifs
)

logging.basicConfig(level=logging.WARNING)
alphabet = list("XXARNDCQEGHILKMFPSTWYV-")

def handle_pre_run_return(batch, binder_chain_num, alphabet):
    """Handle return values from pre-run phase"""
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
    }
    
    best_seq = extract_sequence_from_batch(batch, binder_chain_num, alphabet)
    
    return (
        batch["res_type"].detach().cpu().numpy(),
        best_seq,
        predict_args
    )


def create_apo_config(data, binder_chain_num, motif_scaffolding):
    """Create apo configuration from holo data"""
    data_apo = copy.deepcopy(data)
    data_apo.pop("constraints", None)
    
    if not motif_scaffolding:
        data_apo.pop("templates", None)
    
    data_apo["sequences"] = [data_apo["sequences"][binder_chain_num]]
    return data_apo


def get_boltz_model(
    checkpoint: Optional[str] = None,
    predict_args=None,
    device: Optional[str] = None,
    model_version: str = "boltz2",
    grad_enabled=True,
    no_potentials = True
) -> Boltz2:
    torch.set_grad_enabled(grad_enabled)
    # TF32 matmul is env-controlled (default "highest"=full FP32, unchanged behavior).
    # Set BOLTZDESIGN_MATMUL_PREC=high to enable TF32 tensor cores (~1.47x/step, no quality
    # loss per BoltzHunter). Only affects processes started after the env var is set.
    torch.set_float32_matmul_precision(os.environ.get("BOLTZDESIGN_MATMUL_PREC", "highest"))
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.638  # Default value
    steering_args = BoltzSteeringParams()



    if no_potentials:
        steering_args.fk_steering = False
        steering_args.contact_guidance_update = False
        steering_args.physical_guidance_update = False

    pairformer_args = (
        PairformerArgsV2() if model_version == "boltz2" else PairformerArgs()
    )
    pairformer_args.v2 = True if model_version == "boltz2" else False
    pairformer_args.activation_checkpointing = True
    # MC-dropout: the design loop already runs the model in train() mode, but the
    # checkpoint sets dropout=0.0, so train mode adds no stochasticity -> every seed
    # funnels to the same fold. BOLTZDESIGN_MC_DROPOUT>0 injects dropout into the
    # Pairformer so each design step sees a slightly different network -> diverse
    # gradients -> escape the shared basin. Default 0.0 = unchanged behavior.
    _mcd = float(os.environ.get("BOLTZDESIGN_MC_DROPOUT", "0.0"))
    if _mcd > 0:
        pairformer_args.dropout = _mcd
        print(f"[MC-dropout] pairformer dropout set to {_mcd}")

    msa_args = MSAModuleArgs(
        subsample_msa=True, num_subsampled_msa=1024, use_paired_feature=True
    )
    msa_args.activation_checkpointing = True

    model_class = Boltz2 if model_version == "boltz2" else Boltz1
    if model_version == "boltz2":
        model_module = model_class.load_from_checkpoint(
            checkpoint,
            strict=False,
            predict_args=predict_args,
            map_location=device,
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
            structure_prediction_training=True,
            no_msa=False,
            no_atom_encoder=False,
            use_templates=True,
            use_templates_v2=True,
            use_trifast=False,
            max_parallel_samples=1,
            steering_args=asdict(steering_args),
            pairformer_args=asdict(pairformer_args),
            msa_args=asdict(msa_args),
        )
    elif model_version == "boltz1":
        model_module = Boltz1.load_from_checkpoint(
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


def _bd_make_optimizer(params, lr, default_factory):
    """env-gated optimizer override for the optimizer-comparison experiment.

    BOLTZDESIGN_OPT in {sgd, sgd_mom, adamw}; unset -> default_factory() (current
    behavior, unchanged). BOLTZDESIGN_MOMENTUM sets SGD momentum (default 0.9 for
    sgd_mom); BOLTZDESIGN_NESTEROV=1 enables Nesterov. Applied at ALL three design
    stages so the whole schedule uses one optimizer (stages 2/3 otherwise hardcode SGD).
    """
    opt = os.environ.get("BOLTZDESIGN_OPT")
    if not opt:
        return default_factory()
    opt = opt.lower()
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if opt in ("sgd", "sgd_mom"):
        mom = float(os.environ.get("BOLTZDESIGN_MOMENTUM", "0.9" if opt == "sgd_mom" else "0.0"))
        nesterov = os.environ.get("BOLTZDESIGN_NESTEROV", "0") == "1" and mom > 0
        return torch.optim.SGD(params, lr=lr, momentum=mom, nesterov=nesterov)
    raise ValueError(f"unknown BOLTZDESIGN_OPT={opt!r} (use sgd|sgd_mom|adamw)")


def boltz_hallucination(
    boltz_model,
    boltz_model_version,
    yaml_path,
    ccd_lib,
    ccd_path,
    length=100,
    binder_chain="",
    design_algorithm="3stages",
    recycling_steps=0,
    pre_iteration=20,
    soft_iteration=50,
    soft_iteration_1=50,
    soft_iteration_2=25,
    temp_iteration=50,
    hard_iteration=10,
    semi_greedy_steps=0,
    learning_rate=0.1,
    learning_rate_pre=0.1,
    learning_rate_end=None,   # if set, anneal base lr learning_rate -> this over the run
                              # (on top of the built-in temp lr_scale). None = fixed lr.
    inter_chain_cutoff=21.0,
    intra_chain_cutoff=14.0,
    num_inter_contacts=2,
    num_intra_contacts=4,
    e_soft=0.8,
    e_soft_1=0.8,
    e_soft_2=1.0,
    alpha=2.0,
    pre_run=False,
    set_train=True,
    disconnect_feats=False,
    disconnect_pairformer=False,
    mask_ligand=False,
    distogram_only=False,
    input_res_type=False,
    non_protein_target=False,
    increasing_contact_over_itr=False,
    loss_scales=None,
    optimize_contact_per_binder_pos=False,
    pocket_conditioning=False,
    pocket_loss=True,
    msa_max_seqs=4096,
    optimizer_type="SGD",
    noise_scaling=0.1,
    save_trajectory=False,
    motif_scaffolding=False,
    motifs=None,
    shifted_motifs=None,
    shifted_fix_motif_pos=None,
    omit_aa_types="C",
    gpu_id=0,
    grad_noise=0.0,
):


    predict_args = {
        "recycling_steps": recycling_steps,  # Default value
        "sampling_steps": 200,  # Default value
        "diffusion_samples": 1,  # Default value
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    boltz_model.predict_args = predict_args

    with yaml_path.open("r") as file:
        data = yaml.safe_load(file)

    if motif_scaffolding and shifted_motifs:
        seq_list = ["X"] * length
        orig_seq = data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"]
        for m_ in shifted_motifs:
            # Extract motif segment and place at shifted position
            m_seq = orig_seq[m_['start_pos'] : m_['end_pos'] + 1]
            seq_list[m_['shifted_start'] : m_['shifted_end']] = list(m_seq)
        data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = "".join(seq_list)
    else:
        data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = "X" * length

    if pocket_loss and pocket_conditioning:
        if "constraints" in data:
            contacts = data["constraints"][0]["pocket"]["contacts"]
            contact_positions = [pos for chain, pos in contacts]
            assert len(np.unique([chain for chain, pos in contacts])) == 1, (
                "only one target chain is supported"
            )
            target_chain = np.unique([chain for chain, pos in contacts])[0]
            assert target_chain != binder_chain, (
                "target chain must be different from the binder chain"
            )
            print("contact_positions", contact_positions)
            print("--------------------------------")
        else:
            print("no pocket constraints")
            contact_positions = None
    else:
        contact_positions = None

    name = yaml_path.stem
    print("data", data)
    target = parse_boltz_schema(
        name,
        data,
        ccd_lib,
        ccd_path,
        boltz_2=True if boltz_model_version == "boltz2" else False,
    )
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    boltz_model.train() if set_train else boltz_model.eval()
    print(f"set in {'train' if set_train else 'eval'} mode")
    # Optimization: we only optimize res_type_logits, never the model weights, so
    # computing/accumulating weight gradients in backward() is pure waste. Freezing
    # the model params (requires_grad=False) skips the weight-grad matmuls and drops
    # activations only needed for them. The logit-gradient is numerically identical.
    # env-gated so the default path is unchanged.
    if os.environ.get("BOLTZDESIGN_FREEZE_WEIGHTS", "1") == "1":
        _nfrozen = 0
        for _p in boltz_model.parameters():
            if _p.requires_grad:
                _p.requires_grad_(False)
                _nfrozen += 1
        print(f"[freeze-weights] froze {_nfrozen} boltz param tensors (weight grads off)")
    # env-gated reproducibility: lets us verify freeze vs no-freeze from an identical
    # init (same Gumbel logit init + same forward RNG stream). Off by default.
    _seed = os.environ.get("BOLTZDESIGN_SEED")
    if _seed is not None:
        _s = int(_seed)
        random.seed(_s); np.random.seed(_s)
        torch.manual_seed(_s); torch.cuda.manual_seed_all(_s)
        print(f"[seed] global seed set to {_s}")
    # "keep soft" lever (over-optimization study): floor the softmax temperature endpoint
    # so confident positions don't freeze into a sharp/adversarial minimum. 0.01 = default.
    _min_temp = float(os.environ.get("BOLTZDESIGN_MIN_TEMP", "0.01"))
    if _min_temp != 0.01:
        print(f"[min-temp] temperature endpoint floored at {_min_temp}")

    def get_batch(
        target,
        max_seqs=0,
        length=100,
        pocket_conditioning=False,
        keep_record=False,
        boltz_model_version=None,
    ):
        target_id = target.record.id
        structure = target.structure

        coords = np.array([(atom["coords"],) for atom in structure.atoms], dtype=Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)

        if boltz_model_version == "boltz2":
            structure = StructureV2(
                atoms=structure.atoms,
                bonds=structure.bonds,
                residues=structure.residues,
                chains=structure.chains,
                interfaces=structure.interfaces,
                mask=structure.mask,
                coords=coords,
                ensemble=ensemble,
            )

        elif boltz_model_version == "boltz1":
            structure = Structure(
                atoms=structure.atoms,
                bonds=structure.bonds,
                residues=structure.residues,
                chains=structure.chains,
                interfaces=structure.interfaces,
                mask=structure.mask,
                connections=structure.connections,
            )

        msas = {}
        for chain in target.record.chains:
            msa_id = chain.msa_id
            if msa_id != -1:
                msa = np.load(msa_id)
                msas[chain.chain_id] = MSA(**msa)

        input = Input(
            structure,
            msas,
            record=target.record,
            residue_constraints=target.residue_constraints,
            templates=target.templates,
            extra_mols=target.extra_mols,
        )

        if boltz_model_version == "boltz2":
            tokenizer = Boltz2Tokenizer()
            featurizer = Boltz2Featurizer()
        elif boltz_model_version == "boltz1":
            tokenizer = BoltzTokenizer()
            featurizer = BoltzFeaturizer()

        tokenized = tokenizer.tokenize(input)

        seed = 42
        random = np.random.default_rng(seed)
        if boltz_model_version == "boltz2":
            molecules = {}
            molecules.update(ccd_lib)
            molecules.update(input.extra_mols)
            mol_names = set(tokenized.tokens["res_name"].tolist())
            mol_names = mol_names - set(molecules.keys())
            molecules.update(load_molecules(ccd_path, mol_names))
        options = target.record.inference_options
        if pocket_conditioning:
            pocket_constraints = options.pocket_constraints
            if boltz_model_version == "boltz2":
                batch = featurizer.process(
                    tokenized,
                    random=random,
                    molecules=molecules,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=max_seqs,
                    pad_to_max_seqs=False,
                    compute_symmetries=False,
                    single_sequence_prop=0.0,
                    compute_frames=True,
                    inference_pocket_constraints=pocket_constraints,
                    compute_constraint_features=True,
                    compute_affinity=False,
                )
            elif boltz_model_version == "boltz1":
                binders, pocket = options.binders, options.pocket
                batch = featurizer.process(
                    tokenized,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=max_seqs,
                    pad_to_max_seqs=False,
                    symmetries={},
                    compute_symmetries=False,
                    inference_binder=binders,
                    inference_pocket=pocket,
                )

        else:
            pocket_constraints = None
            if boltz_model_version == "boltz2":
                batch = featurizer.process(
                    tokenized,
                    random=random,
                    molecules=molecules,
                    training=False,
                    max_atoms=None,
                    max_tokens=None,
                    max_seqs=max_seqs,
                    pad_to_max_seqs=False,
                    compute_symmetries=False,
                    single_sequence_prop=0.0,
                    compute_frames=True,
                    inference_pocket_constraints=pocket_constraints,
                    compute_constraint_features=True,
                    compute_affinity=False,
                )
            else:
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

        if keep_record:
            batch["record"] = target.record

        return batch, structure


    def motif_scaffolding_template(batch, shifted_motifs):
        template_keys = ["template_restype", "template_frame_rot", "template_frame_t", "template_ca", "template_mask"]
        # Initialize with zeros
        for key in template_keys:
            if key in batch:
                orig = batch[key].clone()
                batch[key] = torch.zeros_like(batch[key])
                # Fill only motif regions
                for m_ in shifted_motifs:
                    batch[key][:, :, m_['shifted_start']:m_['shifted_end']] = orig[:, :, m_['shifted_start']:m_['shifted_end']]
        return batch

    batch, structure = get_batch(
        target,
        max_seqs=msa_max_seqs,
        length=length,
        pocket_conditioning=pocket_conditioning,
        boltz_model_version=boltz_model_version,
    )
    batch = {key: value.unsqueeze(0).to(device) for key, value in batch.items()}
    if boltz_model_version == "boltz2":
        batch["msa"] = torch.nn.functional.one_hot(batch["msa"], num_classes=33)

    if motif_scaffolding:
        batch = motif_scaffolding_template(batch, shifted_motifs)

    chain_mask = (batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain]).int()
    chain_indices = torch.where(chain_mask)[1]
    
    # Initialize res_type_logits
    if pre_run:
        batch["res_type_logits"] = batch["res_type"].clone().detach().to(device).float()
        omit_mask = get_omit_mask(alphabet, omit_aa_types, device)
        
        if motif_scaffolding and shifted_motifs:
            # Generate mask for ALL scaffold positions (True = Scaffold, False = Motif)
            is_scaffold = torch.ones(len(chain_indices), dtype=torch.bool, device=device)
            
            for m_ in shifted_motifs:
                m_range = torch.arange(m_['shifted_start'], m_['shifted_end'], device=device)
                is_scaffold[m_range] = False
            
            # If user provided specific indices to fix within motifs
            if shifted_fix_motif_pos is not None:
                is_scaffold[shifted_fix_motif_pos] = False

            # Apply noise ONLY to scaffold (non-motif) positions
            batch["res_type_logits"][:, chain_indices[is_scaffold], :] = (
                noise_scaling * torch.softmax(
                    torch.distributions.Gumbel(0, 1).sample(
                        batch["res_type"][:, chain_indices[is_scaffold], :].shape
                    ).to(device) - omit_mask,
                    dim=-1,
                )
            )
        else:
            batch["res_type_logits"][
                batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain], :
            ] = noise_scaling * torch.softmax(
                torch.distributions.Gumbel(0, 1).sample(
                    batch["res_type"][
                        batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain], :
                    ].shape
                ).to(device) - omit_mask,
                dim=-1,
            )
    else:
        batch["res_type_logits"] = torch.from_numpy(input_res_type).to(device)



    if non_protein_target:
        batch = update_batch_msa(batch, CHAIN_TO_NUMBER[binder_chain], non_protein_target, device)
        batch["msa_paired"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["deletion_value"] = torch.zeros(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["has_deletion"] = torch.full(
            (batch["res_type"].shape[0], 1, batch["res_type"].shape[1]), False
        ).to(device)
        batch["msa_mask"] = torch.ones(
            batch["res_type"].shape[0], 1, batch["res_type"].shape[1]
        ).to(device)
        batch["deletion_mean"] = torch.zeros(batch["deletion_mean"].shape).to(device)
        batch["res_type"] = batch["res_type"].float()

    if batch["res_type_logits"].dtype != torch.float32:
        batch["res_type_logits"] = batch["res_type_logits"].float()

    batch["res_type_logits"].requires_grad = True
    _lr0 = learning_rate_pre if pre_run else learning_rate
    def _default_opt():
        return (
            torch.optim.AdamW([batch["res_type_logits"]], lr=_lr0)
            if optimizer_type == "AdamW"
            else torch.optim.SGD([batch["res_type_logits"]], lr=_lr0)
        )
    optimizer = _bd_make_optimizer([batch["res_type_logits"]], _lr0, _default_opt)

    def norm_seq_grad(grad, chain_mask):
        chain_mask = chain_mask.bool()
        masked_grad = grad[:, chain_mask.squeeze(0), :]
        eff_L = (masked_grad.pow(2).sum(-1, keepdim=True) > 0).sum(-2, keepdim=True)
        gn = masked_grad.norm(dim=(-1, -2), keepdim=True)
        return grad * torch.sqrt(torch.tensor(eff_L)) / (gn + 1e-7)

    best_batch = None
    first_step_best_batch = None

    plots = []
    distogram_history = []
    sequence_history = []
    loss_history = []
    lr_history = []
    con_loss_history = []
    i_con_loss_history = []
    plddt_loss_history = []
    mask = torch.ones_like(batch["res_type_logits"])
    mask[batch["entity_id"] != CHAIN_TO_NUMBER[binder_chain], :] = 0
    if motif_scaffolding and shifted_motifs:
        binder_mask = batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain]
        motif_mask = torch.zeros_like(binder_mask)
        for m_ in shifted_motifs:
            motif_mask[:, m_['shifted_start'] : m_['shifted_end']] = True
        if shifted_fix_motif_pos is not None:
            motif_mask[:, shifted_fix_motif_pos] = True
            
        # Final step: Set mask to 0 for any position identified as a motif
        mask[binder_mask & motif_mask] = 0
    mid_points = torch.linspace(2, 22, 64).to(device)

    if contact_positions is not None:
        mask_hotspot = torch.zeros_like(batch["entity_id"])
        target_chain_mask = batch["entity_id"] == CHAIN_TO_NUMBER[target_chain]
        target_chain_start = torch.where(target_chain_mask)[1][0]
        adjusted_contact_positions = [
            pos + target_chain_start for pos in contact_positions
        ]
        mask_hotspot[:, adjusted_contact_positions] = 1

    def calculate_contact_losses(pdist, mid_pts, chain_mask, chain_1b, 
                                 num_inter, num_intra, inter_cutoff, intra_cutoff,
                                 optimize_per_pos=False, contact_positions=None,
                                 mask_hotspot=None, num_optimizing_pos=1,
                                 increasing_contact=False, pre_run=False):
        """Unified contact loss calculation"""
        con_loss = get_con_loss(pdist, mid_pts, num=num_intra, seqsep=9,
                               cutoff=intra_cutoff, mask_1d=chain_mask, mask_1b=chain_mask)
        
        if optimize_per_pos:
            if contact_positions is not None:
                chain_1b_pos = chain_1b * mask_hotspot
                chain_1b_neg = chain_1b * (1 - mask_hotspot)
            else:
                chain_1b_pos = chain_1b
                chain_1b_neg = None
            
            num_pos = 0 if (pre_run and increasing_contact) else num_optimizing_pos
            
            i_con_loss = get_con_loss(pdist, mid_pts, num=num_inter, seqsep=0,
                                      num_pos=num_pos if increasing_contact else float('inf'),
                                      cutoff=inter_cutoff, mask_1d=chain_mask, 
                                      mask_1b=chain_1b_pos)
            
            if contact_positions is not None and chain_1b_neg is not None:
                negative_loss = get_con_loss(pdist, mid_pts, num=num_inter, seqsep=0,
                                             num_pos=num_pos if increasing_contact else float('inf'),
                                             cutoff=inter_cutoff, mask_1d=chain_mask, 
                                             mask_1b=chain_1b_neg)
                i_con_loss = i_con_loss - negative_loss
        else:
            i_con_loss = get_con_loss(pdist, mid_pts, num=num_inter, seqsep=0,
                                      cutoff=inter_cutoff, mask_1d=chain_1b, 
                                      mask_1b=chain_mask)
        
        return con_loss, i_con_loss

    def design(
        batch,
        iters=None,
        soft=0.0,
        e_soft=None,
        step=1.0,
        e_step=None,
        temp=1.0,
        e_temp=None,
        hard=0.0,
        e_hard=None,
        num_optimizing_binder_pos=1,
        e_num_optimizing_binder_pos=1,
        learning_rate=1.0,
        inter_chain_cutoff=21.0,
        intra_chain_cutoff=14.0,
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
        binder_chain="A",
        non_protein_target=False,
        increasing_contact_over_itr=False,
        optimize_contact_per_binder_pos=False,
        num_inter_contacts=2,
        num_intra_contacts=4,
        save_trajectory=False,
        motif_scaffolding=False,
        shifted_motifs=None,
        shifted_fix_motif_pos=None,
    ):
        def get_model_loss(
            batch,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
            pre_run=False,
            mask_ligand=False,
            distogram_only=False,
            predict_args=None,
            loss_scales=None,
            binder_chain="A",
            increasing_contact_over_itr=False,
            optimize_contact_per_binder_pos=False,
            num_inter_contacts=2,
            num_intra_contacts=4,
            num_optimizing_binder_pos=1,
            inter_chain_cutoff=21.0,
            intra_chain_cutoff=14.0,
            save_trajectory=False,
        ):
            traj_coords = None
            traj_plddt = None

            # Handle masking first if needed
            if pre_run and mask_ligand:
                batch["token_pad_mask"][
                    batch["entity_id"] != CHAIN_TO_NUMBER[binder_chain]
                ] = 0
                masked_token_to_rep = torch.ones_like(batch["token_to_rep_atom"])
                masked_token_to_rep[
                    batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain], :
                ] = 0
                masked_token_to_rep_index = torch.nonzero(
                    batch["token_to_rep_atom"] * masked_token_to_rep, as_tuple=True
                )[2]
                batch["atom_pad_mask"][:, masked_token_to_rep_index] = 0

            # Common arguments for get_distogram_confidence
            confidence_args = {
                "recycling_steps": predict_args["recycling_steps"],
                "num_sampling_steps": predict_args["sampling_steps"],
                "multiplicity_diffusion_train": 1,
                "diffusion_samples": predict_args["diffusion_samples"],
                "run_confidence_sequentially": True,
                "disconnect_feats": disconnect_feats,
                "disconnect_pairformer": disconnect_pairformer,
            }

            if save_trajectory:
                # Get model output with trajectory info
                dict_out = boltz_model.get_distogram_confidence(
                    batch, **confidence_args
                )
                traj_coords = dict_out["sample_atom_coords"][0].detach().cpu().numpy()
                traj_plddt = dict_out["plddt"][0].detach().cpu().numpy()
            else:
                # Get model output without trajectory
                if pre_run or distogram_only:
                    dict_out, s, z, s_inputs = boltz_model.get_distogram(batch)
                else:
                    dict_out = boltz_model.get_distogram_confidence(
                        batch, **confidence_args
                    )

            pdist = dict_out["pdistogram"].squeeze(-2)  # Shape: BxLxLxD
            mid_pts = get_mid_points(pdist).to(device)

            # Calculate contact losses using the new unified function
            chain_1b = 1 - chain_mask
            if contact_positions is not None:
                chain_1b = chain_1b * mask_hotspot
            
            con_loss, i_con_loss = calculate_contact_losses(
                pdist, mid_pts, chain_mask, chain_1b,
                num_inter_contacts, num_intra_contacts,
                inter_chain_cutoff, intra_chain_cutoff,
                optimize_per_pos=optimize_contact_per_binder_pos,
                contact_positions=contact_positions,
                mask_hotspot=mask_hotspot if contact_positions else None,
                num_optimizing_pos=num_optimizing_binder_pos,
                increasing_contact=increasing_contact_over_itr,
                pre_run=pre_run
            )

            mask_2d = chain_mask[:, :, None] * chain_mask[:, None, :]
            helix_loss = get_helix_loss(
                pdist, mid_pts, offset=None, mask_2d=mask_2d, binary=True
            )

            if pre_run and mask_ligand:
                losses = {"con_loss": con_loss, "helix_loss": helix_loss}
            else:
                losses = {
                    "con_loss": con_loss,
                    "i_con_loss": i_con_loss,
                    "helix_loss": helix_loss,
                }

            if not pre_run and not distogram_only:
                plddt_loss = get_plddt_loss(dict_out["plddt"], mask_1d=chain_mask)
                pae = (dict_out["pae"] + dict_out["pae"].transpose(-2, -1)) / 2
                chain_1b = 1 - chain_mask
                if contact_positions is not None:
                    chain_1b = chain_1b * mask_hotspot
                i_pae_loss = get_pae_loss(pae, mask_1d=chain_1b, mask_1b=chain_mask)
                pae_loss = get_pae_loss(pae, mask_1d=chain_mask, mask_1b=chain_mask)
                rg_loss, rg = add_rg_loss(
                    dict_out["sample_atom_coords"],
                    batch,
                    length,
                    binder_chain=binder_chain,
                )

                losses.update(
                    {
                        "plddt_loss": plddt_loss,
                        "i_pae_loss": i_pae_loss,
                        "pae_loss": pae_loss,
                        "rg_loss": rg_loss,
                    }
                )

                plddt_loss_history.append(plddt_loss.item())

            bins = mid_points < 8.0
            px = torch.sum(torch.softmax(pdist, dim=-1)[:, :, :, bins], dim=-1)

            if loss_scales is None:
                print("loss_scales is None, using default loss scales")
                loss_scales = {
                    "con_loss": 1.0,
                    "i_con_loss": 1.0,
                    "helix_loss": random.uniform(-0.4, 0.0),
                    "plddt_loss": 0.1,
                    "pae_loss": 0.4,
                    "i_pae_loss": 0.1,
                    "rg_loss": 0.0,
                }

            # Calculate total loss and print individual losses
            print(f"loss_scales: {loss_scales}")
            total_loss = sum(loss * loss_scales[name] for name, loss in losses.items())
            loss_str = [f"{k}:{v.item():.2f}" for k, v in losses.items()]
            plots.append(px[0].detach().cpu().numpy())
            loss_history.append(total_loss.item())
            i_con_loss_history.append(i_con_loss.item())
            con_loss_history.append(con_loss.item())
            distogram_history.append(px[0].detach().cpu().numpy())
            sequence_history.append(
                batch["res_type"][0, :, 2:22].detach().cpu().numpy()
            )

            return (
                total_loss,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                distogram_history,
                sequence_history,
                plddt_loss_history,
                loss_str,
                traj_coords,
                traj_plddt,
            )

        def update_sequence(
            opt, batch, mask, alpha=2.0, non_protein_target=False, binder_chain="A"
        ):
            omit_mask = get_omit_mask(alphabet, omit_aa_types, device)
            batch["logits"] = alpha * batch["res_type_logits"]
            X = batch["logits"] - omit_mask
            batch["soft"] = torch.softmax(X / opt["temp"], dim=-1)
            # item 3: keep dL/d(prob) available for the direct-probability update.
            if os.environ.get("BOLTZDESIGN_PROB_OPT") == "1":
                batch["soft"].retain_grad()
            batch["hard"] = torch.zeros_like(batch["soft"]).scatter_(
                -1, batch["soft"].max(dim=-1, keepdim=True)[1], 1.0
            )
            batch["hard"] = (batch["hard"] - batch["soft"]).detach() + batch["soft"]
            batch["pseudo"] = (
                opt["soft"] * batch["soft"]
                + (1 - opt["soft"]) * batch["res_type_logits"]
            )
            batch["pseudo"] = (
                opt["hard"] * batch["hard"] + (1 - opt["hard"]) * batch["pseudo"]
            )
            batch["res_type"] = batch["pseudo"] * mask + batch["res_type_logits"] * (
                1 - mask
            )

            batch = update_batch_msa(batch, CHAIN_TO_NUMBER[binder_chain], non_protein_target, device)
            return batch

        m = {
            "soft": [soft, e_soft],
            "temp": [temp, e_temp],
            "hard": [hard, e_hard],
            "step": [step, e_step],
            "num_optimizing_binder_pos": [
                num_optimizing_binder_pos,
                e_num_optimizing_binder_pos,
            ],
        }
        m = {k: [s, (s if e is None else e)] for k, (s, e) in m.items()}

        opt = {}
        traj_coords_list = []
        traj_plddt_list = []
        for i in range(iters):
            start_time = time.time()
            for k, (s, e) in m.items():
                if k == "temp":
                    opt[k] = e + (s - e) * (1 - (i) / iters) ** 2
                else:
                    v = s + (e - s) * ((i) / iters)
                    if k == "step":
                        step = v
                    opt[k] = v

            lr_scale = step * ((1 - opt["soft"]) + (opt["soft"] * opt["temp"]))
            num_optimizing_binder_pos = int(opt["num_optimizing_binder_pos"])

            # Optional annealed BASE lr (high early -> low late), on top of the built-in
            # temp lr_scale. Decouples the exploration phase (high lr -> compact folds)
            # from the refinement phase (low lr -> higher confidence). None = fixed lr.
            if learning_rate_end is not None:
                base_lr = learning_rate + (learning_rate_end - learning_rate) * ((i) / iters)
            else:
                base_lr = learning_rate

            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr * lr_scale

            opt["lr_rate"] = base_lr * lr_scale

            batch = update_sequence(
                opt,
                batch,
                mask,
                non_protein_target=non_protein_target,
                binder_chain=binder_chain,
            )
            (
                total_loss,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                distogram_history,
                sequence_history,
                plddt_loss_history,
                loss_str,
                traj_coords,
                traj_plddt,
            ) = get_model_loss(
                batch,
                plots,
                loss_history,
                i_con_loss_history,
                con_loss_history,
                plddt_loss_history,
                distogram_history,
                sequence_history,
                pre_run,
                mask_ligand,
                distogram_only,
                predict_args,
                loss_scales,
                binder_chain,
                increasing_contact_over_itr,
                optimize_contact_per_binder_pos=optimize_contact_per_binder_pos,
                num_inter_contacts=num_inter_contacts,
                num_intra_contacts=num_intra_contacts,
                num_optimizing_binder_pos=num_optimizing_binder_pos,
                inter_chain_cutoff=inter_chain_cutoff,
                intra_chain_cutoff=intra_chain_cutoff,
                save_trajectory=save_trajectory,
            )
            traj_coords_list.append(traj_coords)
            traj_plddt_list.append(traj_plddt)
            print("total_loss: ", total_loss.item())
            total_loss.backward()

            # item 3: direct-probability (mirror-descent / exponentiated-gradient) update.
            # The default update uses dL/d(logits) = J_softmax . dL/d(prob), whose softmax
            # Jacobian attenuates confident positions (vanishing grad once a residue is
            # ~decided). Overriding the logit grad with dL/d(prob) itself lets confident
            # positions keep moving. The subsequent masking/normalization still applies.
            # env-gated; unset -> unchanged.
            if os.environ.get("BOLTZDESIGN_PROB_OPT") == "1":
                _soft = batch.get("soft", None)
                if (_soft is not None and getattr(_soft, "grad", None) is not None
                        and batch["res_type_logits"].grad is not None):
                    batch["res_type_logits"].grad = _soft.grad.clone()

            if batch["res_type_logits"].grad is not None:
                if motif_scaffolding and shifted_motifs:
                    chain_indices = torch.where(chain_mask)[1]
                    
                    # Identify all positions that are NOT scaffolds (i.e., are motifs)
                    is_motif = torch.zeros(len(chain_indices), dtype=torch.bool, device=device)
                    for m_ in shifted_motifs:
                        is_motif[m_['shifted_start']:m_['shifted_end']] = True
                    
                    if shifted_fix_motif_pos is not None:
                         is_motif[shifted_fix_motif_pos] = True

                    # Zero out gradients for all motif positions to keep them fixed
                    batch["res_type_logits"].grad[:, chain_indices[is_motif], :] = 0

                batch["res_type_logits"].grad[
                    batch["entity_id"] != CHAIN_TO_NUMBER[binder_chain], :
                ] = 0
                
                omit_indices = [i for i in range(len(alphabet)) if alphabet[i] in omit_aa_types]
                non_protein_indices = [0, 1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                batch["res_type_logits"].grad[..., omit_indices + non_protein_indices] = 0
                
                batch["res_type_logits"].grad = norm_seq_grad(
                    batch["res_type_logits"].grad, chain_mask
                )

                # SGLD-style gradient noise: inject annealed Gaussian noise into the
                # binder-position gradients to escape sharp local minima (e.g. the
                # helix basin the ligand-masked warm-up locks boltz2 into). Noise is
                # scaled to the gradient's own magnitude (dimensionless grad_noise)
                # and annealed linearly to 0 over the stage. Restricted to the binder
                # chain + allowed AA columns, mirroring the gradient masks above.
                if grad_noise and grad_noise > 0:
                    noise_scale = grad_noise * max(0.0, 1.0 - i / max(iters, 1))
                    if noise_scale > 0:
                        g = batch["res_type_logits"].grad
                        binder_rows = (
                            batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain]
                        )
                        gstd = g[binder_rows].std() + 1e-8
                        noise = torch.randn_like(g) * (noise_scale * gstd)
                        noise[~binder_rows, :] = 0
                        noise[..., omit_indices + non_protein_indices] = 0
                        batch["res_type_logits"].grad = g + noise

                optimizer.step()
                optimizer.zero_grad()
                current_lr = optimizer.param_groups[0]["lr"]
                end_time = time.time()
                print(
                    f"Epoch {i}: lr: {current_lr:.3f}, soft: {opt['soft']:.2f}, hard: {opt['hard']:.2f}, temp: {opt['temp']:.2f}, total loss: {total_loss.item():.2f}, {loss_str}"
                )
                print(f"Time for iteration {i}: {end_time - start_time:.2f} seconds")

        return (
            batch,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
            traj_coords_list,
            traj_plddt_list,
        )
    def run_design_stage(
        batch,
        stage_name,
        iters,
        stage_params,
        common_params,
        plots,
        loss_history,
        i_con_loss_history,
        con_loss_history,
        plddt_loss_history,
        distogram_history,
        sequence_history,
    ):
        """Helper function to run a single design stage with specific parameters"""
        print("-" * 100)
        print(stage_name)
        print("-" * 100)
        
        # Merge stage-specific params with common params
        design_params = {**common_params, **stage_params, "iters": iters}
        
        return design(
            batch,
            mask=mask,
            chain_mask=chain_mask,
            plots=plots,
            loss_history=loss_history,
            i_con_loss_history=i_con_loss_history,
            con_loss_history=con_loss_history,
            plddt_loss_history=plddt_loss_history,
            distogram_history=distogram_history,
            sequence_history=sequence_history,
            **design_params
        )

    # Common parameters used across all design stages
    common_design_params = {
        "learning_rate": learning_rate,
        "length": length,
        "pre_run": pre_run,
        "distogram_only": distogram_only,
        "predict_args": predict_args,
        "loss_scales": loss_scales,
        "binder_chain": binder_chain,
        "increasing_contact_over_itr": increasing_contact_over_itr,
        "optimize_contact_per_binder_pos": optimize_contact_per_binder_pos,
        "non_protein_target": non_protein_target,
        "inter_chain_cutoff": inter_chain_cutoff,
        "intra_chain_cutoff": intra_chain_cutoff,
        "num_inter_contacts": num_inter_contacts,
        "num_intra_contacts": num_intra_contacts,
        "save_trajectory": save_trajectory,
        "motif_scaffolding": motif_scaffolding,
        "shifted_motifs": shifted_motifs,
        "shifted_fix_motif_pos": shifted_fix_motif_pos,
    }

    if pre_run:
        stage_params = {"soft": 1.0, "learning_rate": learning_rate_pre, "mask_ligand": mask_ligand}
        (
            batch,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
            traj_coords_list,
            traj_plddt_list,
        ) = run_design_stage(
            batch,
            "Pre-run stage",
            pre_iteration,
            stage_params,
            common_design_params,
            plots,
            loss_history,
            i_con_loss_history,
            con_loss_history,
            plddt_loss_history,
            distogram_history,
            sequence_history,
        )
    else:
        if design_algorithm == "3stages":
            # Stage 1: logits to softmax
            stage1_params = {
                "e_soft": e_soft,
                "num_optimizing_binder_pos": 1,
                "e_num_optimizing_binder_pos": 8,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list1, traj_plddt_list1,
            ) = run_design_stage(
                batch, f"logits to softmax(T={e_soft})", soft_iteration,
                stage1_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            # Stage 2: softmax temperature annealing
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch["res_type_logits"] = new_logits
            optimizer = _bd_make_optimizer(
                [batch["res_type_logits"]], learning_rate,
                lambda: torch.optim.SGD([batch["res_type_logits"]], lr=learning_rate))
            
            stage2_params = {
                "soft": 1.0,
                "temp": 1.0,
                "e_temp": _min_temp,
                "num_optimizing_binder_pos": 8,
                "e_num_optimizing_binder_pos": 12,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list2, traj_plddt_list2,
            ) = run_design_stage(
                batch, "softmax(T=1) to softmax(T=0.01)", temp_iteration,
                stage2_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            # Stage 3: hard selection
            stage3_params = {
                "soft": 1.0,
                "hard": 1.0,
                "temp": _min_temp,
                "num_optimizing_binder_pos": 12,
                "e_num_optimizing_binder_pos": 16,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list3, traj_plddt_list3,
            ) = run_design_stage(
                batch, "hard", hard_iteration,
                stage3_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            traj_coords_list = (
                traj_coords_list1 + traj_coords_list2 + traj_coords_list3
                if save_trajectory else []
            )
            traj_plddt_list = (
                traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3
                if save_trajectory else []
            )

        elif design_algorithm == "3stages_extra":
            # Stage 1a: First soft stage
            stage1a_params = {
                "e_soft": e_soft_1,
                "num_optimizing_binder_pos": 1,
                "e_num_optimizing_binder_pos": 8,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list1, traj_plddt_list1,
            ) = run_design_stage(
                batch, f"logits to softmax(T={e_soft_1})", soft_iteration_1,
                stage1a_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            # Stage 1b: Second soft stage
            stage1b_params = {
                "e_soft": e_soft_2,
                "num_optimizing_binder_pos": 1,
                "e_num_optimizing_binder_pos": 8,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list2, traj_plddt_list2,
            ) = run_design_stage(
                batch, f"logits to softmax(T={e_soft_2})", soft_iteration_2,
                stage1b_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            # Stage 2: Temperature annealing
            print("set res_type_logits to logits")
            new_logits = (alpha * batch["res_type_logits"]).clone().detach().requires_grad_(True)
            batch["res_type_logits"] = new_logits
            optimizer = _bd_make_optimizer(
                [batch["res_type_logits"]], learning_rate,
                lambda: torch.optim.SGD([batch["res_type_logits"]], lr=learning_rate))
            
            stage2_params = {
                "soft": 1.0,
                "temp": 1.0,
                "e_temp": _min_temp,
                "num_optimizing_binder_pos": 8,
                "e_num_optimizing_binder_pos": 12,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list3, traj_plddt_list3,
            ) = run_design_stage(
                batch, "softmax(T=1) to softmax(T=0.01)", temp_iteration,
                stage2_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            # Stage 3: Hard selection
            stage3_params = {
                "soft": 1.0,
                "hard": 1.0,
                "temp": _min_temp,
                "num_optimizing_binder_pos": 12,
                "e_num_optimizing_binder_pos": 16,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list4, traj_plddt_list4,
            ) = run_design_stage(
                batch, "hard", hard_iteration,
                stage3_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )

            traj_coords_list = (
                traj_coords_list1 + traj_coords_list2 + traj_coords_list3 + traj_coords_list4
                if save_trajectory else []
            )
            traj_plddt_list = (
                traj_plddt_list1 + traj_plddt_list2 + traj_plddt_list3 + traj_plddt_list4
                if save_trajectory else []
            )

        elif design_algorithm == "logits":
            stage_params = {
                "soft": 0.0,
                "e_soft": 0.0,
            }
            (
                batch, plots, loss_history, i_con_loss_history, con_loss_history,
                plddt_loss_history, distogram_history, sequence_history,
                traj_coords_list, traj_plddt_list,
            ) = run_design_stage(
                batch, "logits", soft_iteration,
                stage_params, common_design_params, plots, loss_history,
                i_con_loss_history, con_loss_history, plddt_loss_history,
                distogram_history, sequence_history,
            )
    def _run_model(boltz_model, batch, predict_args):
        with torch.no_grad():
            boltz_model.predict_args = predict_args
            output = boltz_model.predict_step(batch, batch_idx=0, dataloader_idx=0)
        torch.cuda.empty_cache()
        return output

    if pre_run:
        predict_args = {
            "recycling_steps": 3,  # Default value
            "sampling_steps": 200,  # Default value
            "diffusion_samples": 1,  # Default value
            "write_confidence_summary": True,
            "write_full_pae": True,
            "write_full_pde": False,
        }

        best_logits = batch["res_type_logits"]
        best_seq = "".join(
            [
                alphabet[i]
                for i in torch.argmax(
                    batch["res_type"][
                        batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain], :
                    ],
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            ]
        )
        data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = (
            best_seq
        )

    
        return (
            batch["res_type"].detach().cpu().numpy(),
            plots,
            loss_history,
            distogram_history,
            sequence_history,
            traj_coords_list,
            traj_plddt_list,
        )
    boltz_model.structure_prediction_training = False
    boltz_model.eval()
    print("boltz_model.structure_prediction_training", boltz_model.structure_prediction_training)
    
    if best_batch is None:
        best_batch = first_step_best_batch if first_step_best_batch is not None else batch
    
    predict_args = {
        "recycling_steps": 3,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "max_parallel_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": True,
        "write_full_pde": False,
    }
    
    # Extract best sequence
    best_logits = best_batch["res_type_logits"]
    best_seq = extract_sequence_from_batch(best_batch, CHAIN_TO_NUMBER[binder_chain], alphabet)
    data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = best_seq
    
    # Create apo configuration
    data_apo = copy.deepcopy(data)
    data_apo.pop("constraints", None)
    if not motif_scaffolding:
        data_apo.pop("templates", None)
    data_apo["sequences"] = [data_apo["sequences"][CHAIN_TO_NUMBER[binder_chain]]]
    
    # Update batches with final sequence
    def _update_batches(data, data_apo, boltz_model_version="boltz2"):
        target = parse_boltz_schema(
            name, data, ccd_lib, ccd_path,
            boltz_2=True if boltz_model_version == "boltz2" else False,
        )
        target_apo = parse_boltz_schema(
            name, data_apo, ccd_lib, ccd_path,
            boltz_2=True if boltz_model_version == "boltz2" else False,
        )
        best_batch, best_structure = get_batch(
            target, msa_max_seqs, length,
            keep_record=True, boltz_model_version=boltz_model_version,
        )
        best_batch_apo, best_structure_apo = get_batch(
            target_apo, msa_max_seqs, length,
            keep_record=True, boltz_model_version=boltz_model_version,
        )
        best_batch = {
            key: value.unsqueeze(0).to(device) if key != "record" else value
            for key, value in best_batch.items()
        }
        best_batch_apo = {
            key: value.unsqueeze(0).to(device) if key != "record" else value
            for key, value in best_batch_apo.items()
        }
        return best_batch, best_batch_apo, best_structure, best_structure_apo
    
    best_batch, best_batch_apo, best_structure, best_structure_apo = _update_batches(
        data, data_apo, boltz_model_version=boltz_model_version
    )
    
    if motif_scaffolding and shifted_motifs:
        best_batch = motif_scaffolding_template(
            best_batch, shifted_motifs
        )
        best_batch_apo = motif_scaffolding_template(
            best_batch_apo, shifted_motifs
        )
    
    # Run initial predictions
    output = _run_model(boltz_model, best_batch, predict_args)
    output_apo = _run_model(boltz_model, best_batch_apo, predict_args)
    
    ##Semi-greedy optimization
    if semi_greedy_steps > 0:
        def _mutate(sequence, best_logits, i_prob):
            """Mutate sequence at position with highest uncertainty"""
            mutated_sequence = list(sequence)
            i = np.random.choice(np.arange(len(i_prob)), p=i_prob / i_prob.sum())
            i_logits = best_logits[:, i]
            i_logits = i_logits - torch.max(i_logits)
            
            omit_mask = get_omit_mask(alphabet, omit_aa_types, device)
            i_X = i_logits - omit_mask
            i_aa = torch.multinomial(torch.softmax(i_X, dim=-1), 1).item()
            mutated_sequence[i] = alphabet[i_aa]
            
            return "".join(mutated_sequence)
        
        prev_sequence = best_seq
        prev_iptm = output["iptm"].detach().cpu().numpy()
        
        print("Best design iptm:", prev_iptm)
        print("Semi-greedy steps:", semi_greedy_steps)
        
        for step in range(semi_greedy_steps):
            confidence_score = []
            mutated_sequence_ls = []
            
            for t in range(10):
                plddt = output["plddt"][
                    best_batch["entity_id"] == CHAIN_TO_NUMBER[binder_chain]
                ]
                i_prob = (
                    np.ones(length) if plddt is None 
                    else torch.maximum(1 - plddt, torch.tensor(0))
                )
                i_prob = (
                    i_prob.detach().cpu().numpy() if torch.is_tensor(i_prob) 
                    else i_prob
                )
                
                mutated_sequence = _mutate(prev_sequence, best_logits, i_prob)
                data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = mutated_sequence
                
                best_batch, _, _, _ = _update_batches(
                    data, data_apo, boltz_model_version=boltz_model_version
                )
                
                if motif_scaffolding and shifted_motifs:
                    best_batch = motif_scaffolding_template(
                        best_batch, shifted_motifs
                    )
                
                output = _run_model(boltz_model, best_batch, predict_args)
                iptm = output["iptm"].detach().cpu().numpy()
                confidence_score.append(iptm)
                mutated_sequence_ls.append(mutated_sequence)
                print(f"Step {step}, Epoch {t}, iptm {iptm[0]:.3f}")
            
            best_id = np.argmax(confidence_score)
            best_iptm = confidence_score[best_id]
            
            if best_iptm > prev_iptm:
                best_seq = mutated_sequence_ls[best_id]
                data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = best_seq
                data_apo["sequences"][0]["protein"]["sequence"] = best_seq
                print(
                    f"Step {step}, Epoch {best_id}, Update sequence, "
                    f"iptm {best_iptm}, previous iptm {prev_iptm}"
                )
                print(f"Update sequence {best_seq}")
                prev_iptm = best_iptm
                prev_sequence = best_seq
            else:
                data["sequences"][CHAIN_TO_NUMBER[binder_chain]]["protein"]["sequence"] = prev_sequence
                data_apo["sequences"][0]["protein"]["sequence"] = prev_sequence
            
            best_batch, best_batch_apo, best_structure, best_structure_apo = (
                _update_batches(data, data_apo, boltz_model_version=boltz_model_version)
            )
            
            if motif_scaffolding and shifted_motifs:
                best_batch = motif_scaffolding_template(
                    best_batch, shifted_motifs
                )
                best_batch_apo = motif_scaffolding_template(
                    best_batch_apo, shifted_motifs
                )
            
            if step == semi_greedy_steps - 1:
                output = _run_model(boltz_model, best_batch, predict_args)
                output_apo = _run_model(boltz_model, best_batch_apo, predict_args)
    
    return (
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
    )

def run_boltz_design(
    boltz_path,
    boltz_model_version,
    main_dir,
    yaml_dir,
    boltz_model,
    ccd_path,
    design_samples=1,
    version_name=None,
    config=None,
    loss_scales=None,
    num_workers=1,
    show_animation=False,
    save_trajectory=False,
    redo_boltz_predict=True,
    gpu_id=0,
):
    """Run Boltz protein design pipeline with cleaner organization"""
    
    # Use default config if none provided
    if config is None:
        config = {
            "recycling_steps": 0,
            "pre_iteration": 30,
            "soft_iteration": 75,
            "soft_iteration_1": 50,
            "soft_iteration_2": 25,
            "temp_iteration": 45,
            "hard_iteration": 5,
            "semi_greedy_steps": 0,
            "learning_rate_pre": 0.2,
            "learning_rate": 0.1,
            "inter_chain_cutoff": 21.0,
            "intra_chain_cutoff": 14.0,
            "num_inter_contacts": 2,
            "num_intra_contacts": 4,
            "e_soft": 0.8,
            "e_soft_1": 0.8,
            "e_soft_2": 1.0,
            "design_algorithm": "3stages",
            "set_train": True,
            "disconnect_feats": True,
            "disconnect_pairformer": False,
            "distogram_only": True,
            "binder_chain": "A",
            "non_protein_target": False,
            "increasing_contact_over_itr": False,
            "mask_ligand": False,
            "optimize_contact_per_binder_pos": False,
            "pocket_conditioning": False,
            "msa_max_seqs": 4096,
            "length_min": 95,
            "length_max": 160,
            "helix_loss_min": -0.6,
            "helix_loss_max": -0.2,
            "optimizer_type": "SGD",
            "noise_scaling": 0.1,
            "max_history_length": 100,
            "motif_scaffolding": None,
            "motifs": None,
            "fix_motif_pos": None,
            "min_motif_gap": 10,
            "fix_motif_gap_to_min": False,
        }
    
    config["gpu_id"] = gpu_id
    version_dir = os.path.join(main_dir, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    # Setup directories
    directories = setup_output_directories(version_dir)
    
    # Load CCD library
    ccd_lib = load_canonicals(os.path.expanduser(ccd_path))
    config["ccd_path"] = os.path.expanduser(ccd_path)
    
    # Save config
    config_path = os.path.join(directories['results_final'], "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Filter config for boltz_hallucination
    filtered_config = {
        k: v for k, v in config.items()
        if k not in ["helix_loss_min", "helix_loss_max", "length_min", "length_max", "motifs", "fix_motif_pos", "min_motif_gap", "fix_motif_gap_to_min"]
    }
    
    # Process each YAML file
    for yaml_path in Path(yaml_dir).glob("*.yaml"):
        if not yaml_path.name.endswith(".yaml"):
            continue
        
        target_binder_input = yaml_path.stem
        
        for itr in range(design_samples):
            aggressive_memory_cleanup()
            
            # Randomize length and motif shift
            config["length"] = random.randint(config["length_min"], config["length_max"])
            filtered_config["length"] = config["length"]
            
            if config.get("motif_scaffolding"):
                shifted_motifs, shifted_fix_motif_pos = shift_motifs(
                    motifs=config["motifs"],
                    length=config["length"],
                    fix_motif_pos=config.get("fix_motif_pos"),
                    min_motif_gap=config.get("min_motif_gap", 10),
                    fix_motif_gap_to_min=config.get("fix_motif_gap_to_min", False)
                )
                
                filtered_config["shifted_motifs"] = shifted_motifs
                filtered_config["shifted_fix_motif_pos"] = shifted_fix_motif_pos
            else:
                filtered_config["shifted_motifs"] = None
                filtered_config["shifted_fix_motif_pos"] = None
            
            loss_scales["helix_loss"] = random.uniform(
                config["helix_loss_min"], config["helix_loss_max"]
            )
            
            # Pre-run warm-up
            print("Pre-run warm up")
            print_memory_usage("before pre-run")
            
            (input_res_type, plots, loss_history, distogram_history, 
             sequence_history, traj_coords_list, traj_plddt_list) = \
                boltz_hallucination(
                    boltz_model, boltz_model_version, yaml_path, ccd_lib,
                    **filtered_config, pre_run=True, input_res_type=False,
                    loss_scales=loss_scales, save_trajectory=save_trajectory
                )
            
            print_memory_usage("after pre-run")
            print("Warm up done")
            
            # Main design
            print_memory_usage("before main design")
            
            (output, output_apo, best_batch, best_batch_apo, best_structure,
             best_structure_apo, distogram_history_2, sequence_history_2,
             loss_history_2, con_loss_history, i_con_loss_history,
             plddt_loss_history, traj_coords_list_2, traj_plddt_list_2,
             structure) = \
                boltz_hallucination(
                    boltz_model, boltz_model_version, yaml_path, ccd_lib,
                    **filtered_config, pre_run=False, input_res_type=input_res_type,
                    loss_scales=loss_scales,save_trajectory=save_trajectory
                )
            
            print_memory_usage("after main design")
            
            # Combine history
            loss_history.extend(loss_history_2)
            distogram_history.extend(distogram_history_2)
            sequence_history.extend(sequence_history_2)
            traj_coords_list.extend(traj_coords_list_2)
            traj_plddt_list.extend(traj_plddt_list_2)
            
            # Process and save results
            process_design_results(
                output, output_apo, best_batch, best_batch_apo,
                best_structure, best_structure_apo, distogram_history,
                sequence_history, loss_history, con_loss_history,
                i_con_loss_history, plddt_loss_history, traj_coords_list,
                traj_plddt_list, structure, config, directories,
                yaml_path, target_binder_input, itr, loss_scales,
                boltz_path, boltz_model_version, alphabet,
                redo_boltz_predict, show_animation, save_trajectory
            )
            
            # Cleanup
            cleanup_iteration(
                output, output_apo, best_batch, best_batch_apo,
                best_structure, best_structure_apo, distogram_history_2,
                sequence_history_2, loss_history_2, con_loss_history,
                i_con_loss_history, plddt_loss_history, traj_coords_list_2,
                traj_plddt_list_2, structure, distogram_history,
                sequence_history, loss_history, traj_coords_list, traj_plddt_list
            )