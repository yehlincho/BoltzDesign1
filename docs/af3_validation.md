# AlphaFold 3 validation

Designs are validated with AlphaFold 3 after the LigandMPNN redesign step. The validator
is **warm and batched**: the model is loaded once and reused across the whole batch, the
target MSA is cached, and the apo prediction is skipped for low-iptm designs. Compared to
launching a container per design this is roughly **4-5x faster end-to-end**, with the same
AF3 model and the same numbers.

## AlphaFold 3 is not bundled

AlphaFold 3 source is licensed **CC BY-NC-SA 4.0**, and its model parameters must be
requested from Google DeepMind. Obtain both yourself:

- code: https://github.com/google-deepmind/alphafold3
- parameters: follow the request process in that repository

Nothing from AlphaFold 3 is redistributed here. This repo only ships the validator that
drives it (`boltzdesign/af3/`).

## Why a separate conda environment

AlphaFold 3 runs on **JAX / Python 3.11**; BoltzDesign runs on **PyTorch / Python 3.10**.
They cannot share one environment, so the pipeline runs the validator as a subprocess in
the AF3 environment (`boltzdesign/af3_driver.py`).

## Setup

`setup.sh` creates the AF3 environment automatically if an AlphaFold 3 checkout is found
at `$AF3_ROOT` (default `~/alphafold3`). To do it manually:

```bash
conda create -p ~/.conda/envs/af3 python=3.11 -y
~/.conda/envs/af3/bin/pip install "jax[cuda12]" pandas numpy gemmi biopython
~/.conda/envs/af3/bin/pip install -e ~/alphafold3     # your own AlphaFold 3 checkout
# place the DeepMind-provided parameters in ~/alphafold3/models
```

## Layout

```
boltzdesign/
├── af3/                  validator package
│   ├── validator.py        AF3Validator: warm/batched, holo + apo, metrics
│   ├── base.py             batching, thresholds, result CSVs
│   ├── msa.py              MSA generation and cache
│   ├── af_utils.py         cif/pdb helpers
│   └── runtime.py          loads AF3 entry points from your AF3 install
└── af3_driver.py         runs inside the af3 env (subprocess entry point)
```

## Options

| flag | default | meaning |
|---|---|---|
| `--run_alphafold` | `True` | run the validation step |
| `--af3_env_python` | `~/.conda/envs/af3/bin/python` | python of the af3 env; or `$AF3_ENV_PYTHON` |
| `--alphafold_dir` | `~/alphafold3` | AlphaFold 3 install; exported to the driver as `$AF3_ROOT` |
| `--af3_num_diffusion_samples` | `1` | diffusion samples per design |

`$AF3_PIPELINE` may point at a module providing `make_model_config`, `ModelRunner`,
`predict_structure`, `write_outputs` and `Input`. This is only needed when a local
AlphaFold 3 has been customised such that its own `run_alphafold.py` no longer matches
the installed `alphafold3` package.

## Outputs

```
<run>/ligandmpnn_cutoff_<n>/
├── 02_af3/
│   ├── af3_validation_results.csv     per-design iptm / iptm_global / plddt / rg / apo rmsd
│   ├── af3_structures/                holo cif + pdb
│   └── af3_structures_apo/            apo cif + pdb
├── 03_af_pdb_success/                 designs passing iptm > 0.5 and plddt > 0.7
│   └── high_iptm_confidence_scores.csv
└── 03_af_pdb_apo/                     matching apo structures
```

Success is gated on the **global** iptm (`iptm_global`) so it matches what the
AlphaFold 3 summary confidences report.
