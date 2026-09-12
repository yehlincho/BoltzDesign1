# Fast AF3 validation (`--fast_validation`)

The validation step can run AF3 through ProteinHunter's **warm, batched** validator
instead of the per-design Docker container. Same AF3 model and weights — the speedup is
pure engineering: the model is loaded **once** and kept warm, the target MSA is cached,
and there is no per-design container start. On production batches this is ~15–25× on the
AF3 step (~4–5× end-to-end, since AF3 dominates wall-clock).

It is **on by default** and **auto-falls back to Docker** if the af3 env / ProteinHunter
is not found, so nothing breaks without the setup below. Force Docker with
`--fast_validation False`.

## Requirements

1. **An `af3` conda env** with AlphaFold 3 + JAX installed, and the AF3 weights at
   `~/alphafold3/models` (same weights the Docker path uses). Follow the official
   [AlphaFold 3 install](https://github.com/google-deepmind/alphafold3).
2. **ProteinHunter** checked out locally (provides `validation/` + `af3_ph/`).

## Setup

```bash
# 1. af3 conda env (AF3 + jax); e.g. created at:
#    ~/ProteinHunter/.conda/envs/af3
conda create -p ~/ProteinHunter/.conda/envs/af3 python=3.11
# ... install alphafold3 + jax[cuda] into it per the AF3 instructions ...

# 2. ProteinHunter repo (validation/ + af3_ph/ live here)
git clone <proteinhunter-repo> ~/ProteinHunter
```

Then just run the pipeline as usual — fast validation is used automatically. Override the
locations if they differ from the defaults:

```bash
python boltzdesign.py ... \
    --fast_validation True \
    --af3_ph_env_python ~/ProteinHunter/.conda/envs/af3/bin/python \
    --proteinhunter_root ~/ProteinHunter
```

## Flags

| flag | default | meaning |
|---|---|---|
| `--fast_validation` | `True` | use warm AF3; auto-fallback to Docker if unavailable |
| `--af3_ph_env_python` | `~/ProteinHunter/.conda/envs/af3/bin/python` | af3 conda env python |
| `--proteinhunter_root` | `~/ProteinHunter` | ProteinHunter checkout (has `validation/`, `af3_ph/`) |
| `--af3_num_diffusion_samples` | `1` | AF3 diffusion samples for the warm path |

## Parity

Same AF3 model → **holo iptm/pLDDT and the success set (`iptm>0.5 & pLDDT>0.7`) reproduce**
the Docker path (verified head-to-head). The only quantity that differs is the **apo-holo
RMSD**, which is an independent single-sample AF3 prediction and is therefore inherently
stochastic (two Docker runs disagree on it too). It matters only for the optional Rosetta
`apo_holo_rmsd` filter on borderline designs.
