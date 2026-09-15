# Changelog
All notable changes to this project will be documented in this file.

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
