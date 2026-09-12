#!/usr/bin/env python
"""
ph_af3_driver.py — runs ProteinHunter's WARM/batched AF3 validator over a directory of
Boltz/AF3 redesigned YAMLs, writing af3_validation_results.csv + holo/apo structures.

This is the AF3-env side of Pairformer's `--fast_validation` path. It runs in
ProteinHunter's af3 conda env (which has alphafold3 + af3_ph), NOT the boltz env.
READ-ONLY reuse of ProteinHunter: it only imports validation.*, modifies nothing there.

Parity note: PH's headline `iptm` is binder-specific (max over chain-pairs); Pairformer's
Docker path reads the GLOBAL iptm from summary_confidences.json, which PH persists as
`iptm_global` in the CSV. Downstream success is gated on iptm_global here.

  python ph_af3_driver.py --yaml_dir <redesigned_yaml_dir> --out_dir <ph_out> \
        --binder_id A --gpu 0 [--num_diffusion_samples 1] [--msa_mode single]
"""
import argparse
import glob
import os
import sys

import yaml


def _parse_yaml(path, binder_id):
    """Return (design_name, binder_seq, target_seq_or_None, ligand_ccd_or_None,
    ligand_smiles_or_None) from one redesigned Boltz/AF3 YAML.

    The chain whose id == binder_id is the binder; every other protein chain is target
    (":"-joined for PH), ligands are CCD (","-joined) or SMILES.
    """
    with open(path) as f:
        doc = yaml.safe_load(f)
    binder_seqs, target_seqs, ccds, smiles = [], [], [], []
    for entry in doc.get("sequences", []):
        if "protein" in entry:
            p = entry["protein"]
            ids = p.get("id", [])
            ids = ids if isinstance(ids, list) else [ids]
            if binder_id in ids:
                binder_seqs.append(p["sequence"])
            else:
                target_seqs.append(p["sequence"])
        elif "ligand" in entry:
            lig = entry["ligand"]
            if lig.get("ccd"):
                ccds.append(lig["ccd"])
            elif lig.get("smiles"):
                smiles.append(lig["smiles"])
    if not binder_seqs:
        raise ValueError(f"{path}: no protein chain with id '{binder_id}' (binder)")
    if len(binder_seqs) > 1:
        # PH validates a single binder chain per design; join is unsupported here.
        raise ValueError(f"{path}: {len(binder_seqs)} chains match binder_id "
                         f"'{binder_id}'; expected exactly 1")
    name = os.path.splitext(os.path.basename(path))[0]
    return (name, binder_seqs[0],
            ":".join(target_seqs) if target_seqs else None,
            ",".join(ccds) if ccds else None,
            smiles[0] if smiles else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--binder_id", default="A")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--proteinhunter_root", default=os.path.expanduser("~/ProteinHunter"))
    ap.add_argument("--num_diffusion_samples", type=int, default=1)
    ap.add_argument("--num_recycles", type=int, default=10)
    ap.add_argument("--msa_mode", default="single",
                    help="single | mmseqs | path to a3m (target MSA)")
    ap.add_argument("--iptm_threshold", type=float, default=0.0,
                    help="apo-skip below this iptm; 0 = always predict apo")
    args = ap.parse_args()

    sys.path.insert(0, args.proteinhunter_root)
    from validation.af3_validator import AF3Validator
    from validation.base import ValidationInput

    yamls = sorted(glob.glob(os.path.join(args.yaml_dir, "*.yaml")))
    if not yamls:
        print(f"[ph_af3_driver] no YAMLs in {args.yaml_dir}", flush=True)
        sys.exit(1)

    names, bseqs = [], []
    target_seq = ccd = smi = None
    for y in yamls:
        n, bseq, tseq, c, s = _parse_yaml(y, args.binder_id)
        names.append(n)
        bseqs.append(bseq)
        # All designs in a campaign share one target; take it from the first and
        # warn (don't fail) if a later YAML disagrees.
        if target_seq is None and ccd is None and smi is None:
            target_seq, ccd, smi = tseq, c, s
        elif (tseq, c, s) != (target_seq, ccd, smi):
            print(f"[ph_af3_driver] WARNING: {n} target differs from batch target; "
                  f"using batch target for the warm run", flush=True)

    print(f"[ph_af3_driver] {len(names)} designs | target_seq={'yes' if target_seq else 'None'} "
          f"| ligand_ccd={ccd} | smiles={'yes' if smi else 'None'}", flush=True)

    vinput = ValidationInput(
        target_sequence=target_seq,
        binder_sequences=bseqs,
        binder_ids=names,
        ligand_ccd=ccd,
        ligand_smiles=smi,
    )
    validator = AF3Validator(
        output_dir=args.out_dir,
        device=str(args.gpu),
        msa_mode=args.msa_mode,
        msa_binder=False,
        num_diffusion_samples=args.num_diffusion_samples,
        num_recycles=args.num_recycles,
    )
    validator.validate_batch(vinput, iptm_threshold=args.iptm_threshold)
    print(f"[ph_af3_driver] done -> {os.path.join(args.out_dir, 'af3_validation_results.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
