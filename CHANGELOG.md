# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]
### Changed
- The design loop now runs its trunk forward under bf16 autocast by default, which is the
  precision upstream Boltz-2 uses for inference (`boltz/main.py:1262`). It had been fp32
  only because `get_distogram` is a fork-only function that bypasses the Lightning trainer
  applying that setting, so nothing ever enabled it. Measured on an A100 80GB: 1.85x per
  iteration on a small-molecule target (FAD, 2.649 -> 1.434 s/iter, -11% memory) and 1.95x
  on a protein target (PDL1, 3.097 -> 1.590 s/iter, -22% memory), about 1.6x end to end
  since prep, scoring and file writing are precision-independent. Raw-design quality did
  not degrade on either target (FAD n=13: holo/apo RMSD 6.01 -> 1.60 A, p=0.0023, holo
  pLDDT 0.749 -> 0.844, p=0.069; PDL1 n=3: flat). Losses stay in fp32 -- the distogram,
  pLDDT, PAE and coordinates are cast back before any softmax, top-k or log-sum-exp, and
  the optimized logits, their gradients and the SGD update were never in the autocast
  region. `BOLTZDESIGN_AUTOCAST=fp32` restores the previous behaviour.
  NOT yet confirmed end to end: the quality evidence is Boltz's own metrics on designs
  before LigandMPNN redesign, on two targets. AF3 strict success (complex pLDDT > 0.7,
  ipAE < 10) across >=20 designs per arm with redesign on is still outstanding, so designs
  made from here on are not precision-comparable to the existing fp32 dataset.
- The final holo/apo scoring calls also run under bf16 autocast, so the precision map now
  matches upstream Boltz-2 everywhere the code is shared. Measured on an identical design
  (`--init_seed` pinned, so both arms scored the same binder): 17.17 -> 15.91 s per design
  (1.08x), with holo complex pLDDT 0.836 -> 0.831, apo 0.836 -> 0.836 and holo/apo RMSD
  0.996 -> 0.962 A. Those shifts are smaller than the run-to-run variation of the
  stochastic sampler at `diffusion_samples=1`, so existing confidence thresholds hold.
  Only 1.08x because the 200-step diffusion sampler is ~90% of that call and boltz pins it
  to fp32 in both codebases; bf16 reaches only the trunk and the confidence head.

### Added
- `BOLTZDESIGN_AUTOCAST=bf16` runs the trunk forward under bf16 autocast, casting the
  distogram, pLDDT, PAE and coordinates back to fp32 before the loss math; default is
  unset, i.e. fp32. The design loop calls the model directly rather than through the
  Lightning trainer, so upstream's `precision="bf16-mixed"` (`boltz/main.py:1262`, Boltz-2
  only) never applied and fp32 was inherited by omission. Measured with `prec_bench.py`
  (boltz2, FAD, 150 aa, seed 42, one A100 80GB per arm): fp32 2.325 s/iter at 6253 MiB,
  TF32 (`BOLTZDESIGN_MATMUL_PREC=high`) 1.580 s/iter (1.47x) at 6255 MiB, bf16
  1.431 s/iter (1.62x) at 5563 MiB. Both remain opt-in: neither is bit-identical, the
  trajectories separate within a few steps, and the resulting designs share only 6-7%
  sequence identity with the fp32 arm, so a yield comparison across many designs is still
  needed before either becomes the default.
- `BOLTZDESIGN_SCORE_KERNELS=1` enables the fused cuequivariance/trifast pair-track
  kernels for the scoring call only, scoped with try/finally so the design loop keeps the
  plain path it needs for backward. Upstream passes `use_kernels=True` for prediction and
  we never did; the packages and an sm_80 card are already present. Default off pending
  measurement -- BoltzHunter previously rejected trifast as ~9% slower at 150 aa.
- `BOLTZDESIGN_NO_CKPT=1` disables activation checkpointing in the Pairformer and MSA
  modules, and `BOLTZDESIGN_NO_MSA_CKPT=1` disables it for the MSA module alone; both stay
  enabled by default. The MSA-only variant was measured on PDL1 at 1.18x per iteration for
  a near-doubling of peak memory (9.2 -> 18.1 GB), and stacked with bf16 it reached only
  1.74x against bf16's own 1.95x, so the recompute is cheaper than the memory pressure. Measured with `ckpt_bench.py` (boltz2, FAD,
  150 aa, one A100 80GB, same seed both arms): checkpointing runs at 2.322 s/iter with a
  6.3 GB peak, and disabling it needs ~79 GB, running out of memory before the first
  iteration finishes (+74.9 GB, 13x). The cost is triangle attention's per-layer
  attention weights (`triangular_attention/primitives.py:191`), about 1.2 GB per block
  across 64 blocks, so the recompute is what keeps the design loop inside 6.3 GB rather
  than a leftover training default. The knob only exists to reproduce that result.

## [3.0.0] - 2026-09-15
### Fixed
- Protein targets were validated single-sequence: AlphaFold 3 never received the target
  MSA or template the design was conditioned on, so the target mis-folded and good
  designs scored as failures. The redesigned YAML's MSA/template are now reused for
  validation (PD-L1 iPTM 0.10 -> 0.82 and 0.20 -> 0.91; ipSAE 0 -> 0.76/0.89).
- `get_target_ids()` was called but never defined after the 2.0 refactor, so every run
  using `--modifications` or `--contact_residues` failed with `NameError`. This broke
  peptide targets carrying PTMs and all pocket-conditioned runs.
- The LigandMPNN step rewrote `LigandMPNN/run_ligandmpnn_logits_config.yaml` in place,
  replacing the portable `${CWD}` template with machine-specific absolute paths on every
  run. The expanded config is now written next to the run instead.
- Design configs were loaded from `configs2/` while `configs/` sat unused, so edits to
  the obvious path silently had no effect.
- Removed a duplicate `np_kabsch` and a shadowed `pr_relax` (the second definition
  silently won; the version with B-factor copying was the live one and is kept).

### Added
- Bundled AlphaFold 3 validator (`boltzdesign/af3/` with `boltzdesign/af3_driver.py`):
  warm and batched, the model is loaded once per batch and target MSAs are cached,
  roughly 4-5x faster end-to-end than one container per design.
- ipSAE (min/max/mean) and binder radius of gyration in `high_iptm_confidence_scores.csv`.
- `setup.sh` builds the AlphaFold 3 environment using AlphaFold 3's own install sequence
  (including `build_data`) and clones its source.
- Design recipe per target type in `boltzdesign/configs/`: Gumbel sequence initialization
  (scale 1.0 for ligand/metal/nucleic/peptide, 2.0 for protein), `e_soft` 1.0,
  learning rate 0.1 (0.2 for protein), `helix_loss` (-0.3, 0.0).

### Changed
- `pre_iteration` defaults to 0. The ligand-masked warm-up seeds elongated helical
  binders that the model cannot refold, which lowers validated success; pass
  `--pre_iteration 30` to reproduce the original protocol.
- `run_examples.py` and `run_examples.ipynb` cover every target type on Boltz-1 and
  Boltz-2, including template mode (Boltz-2 only).

### Removed
- The per-design Docker AlphaFold 3 path, along with `--af3_docker_name`,
  `--af3_database_settings` and `--af3_hmmer_path`. AlphaFold 3 now runs in its own conda
  environment (`--af3_env_python`); AlphaFold 3 itself is not redistributed, so its source
  and DeepMind-issued weights must still be obtained by the user.

## [2.0.0] - 2026-01-26
### Added
- Support for both Boltz-1 and Boltz-2 models in boltzdesign.
- Motif scaffolding. Multiple motifs can be assigned; however, if a motif is too short, there is a possibility that the model will fold into another structure.

### Changed
- Refactored the code for better structure and readability
- Boltz-2 uses a higher default negative helical bias for small-molecule and metal-binding designs.

## [1.0.0] - 2025-04
### Added
- Initial release
