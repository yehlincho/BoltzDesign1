"""BoltzDesign run examples — every target type on both Boltz-2 and Boltz-1.

The design recipe (Gumbel init, learning rate, e_soft, helix range) is applied
automatically from the per-target-type config in boltzdesign/configs/; these examples
only specify the target and the model. Pass any flag explicitly to override the config.

Note: templates are supported by Boltz-2 only (Boltz-1 raises "Templates are not
supported in Boltz 1.0!"), so template examples are boltz2-only.
"""

import subprocess

PDL1 = (
    "FTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQ"
    "LSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNK"
)


def run(*args, model="boltz2", suffix=None):
    """Run boltzdesign.py with the given args on the chosen model."""
    cmd = ["python", "boltzdesign.py", *args,
           "--boltz_model_version", model,
           "--suffix", suffix or model]
    print(" ".join(cmd))
    subprocess.run(cmd)


# --------------------------------------------------------------------------- #
# 1. Protein target — MSA mode (both models)
# --------------------------------------------------------------------------- #
protein_msa = [
    "--name", "8znl", "--target_type", "protein", "--pdb_target_ids", "A",
    "--target_seq", PDL1, "--use_msa", "True", "--msa_max_seqs", "4096",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*protein_msa, model="boltz2")
run(*protein_msa, model="boltz1")

# 1-1. Same, but only repredict designs above an iptm cutoff
run(*protein_msa, "--high_iptm", "True", "--i_ptm_cutoff", "0.7",
    model="boltz2", suffix="boltz2_high_iptm")

# 1-2. Optional: semi-greedy MCMC polish after the gradient design (off by default).
# Each step runs 10 mutation trials scored by Boltz iPTM, so it raises the Boltz
# confidence it optimizes but costs roughly 2.5 min per design.
run(*protein_msa, "--semi_greedy_steps", "1", model="boltz2", suffix="boltz2_semigreedy")

# --------------------------------------------------------------------------- #
# 2. Protein target — template mode (boltz2 only)
# --------------------------------------------------------------------------- #
protein_tmpl = [
    "--name", "8znl", "--target_type", "protein", "--pdb_path", "8znl",
    "--pdb_target_ids", "B", "--use_template", "True",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*protein_tmpl, model="boltz2", suffix="boltz2_template")
# Boltz-1 does not support templates — use MSA mode (example 1) for boltz1.

# --------------------------------------------------------------------------- #
# 3. Small molecule (CCD code) — both models
# --------------------------------------------------------------------------- #
small_molecule = [
    "--name", "7v11", "--target_type", "small_molecule", "--target_seq", "OQO",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*small_molecule, model="boltz2")
run(*small_molecule, model="boltz1")

# 3-1. Small molecule with a template (boltz2 only)
run("--name", "7v11", "--target_type", "small_molecule", "--target_seq", "OQO",
    "--pdb_path", "7v11", "--use_template", "True",
    "--design_samples", "1", "--gpu_id", "0",
    model="boltz2", suffix="boltz2_sm_template")

# --------------------------------------------------------------------------- #
# 4. DNA / RNA — both models
# --------------------------------------------------------------------------- #
dna = [
    "--name", "5zmc", "--target_type", "dna",
    "--target_seq", "GCCCTTCCGGGTCCCC,CGGGGACCCGGAAGGG",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*dna, model="boltz2")
run(*dna, model="boltz1")

# --------------------------------------------------------------------------- #
# 5. Metal — both models
# --------------------------------------------------------------------------- #
metal = [
    "--name", "ZN", "--target_type", "metal", "--target_seq", "ZN",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*metal, model="boltz2")
run(*metal, model="boltz1")

# --------------------------------------------------------------------------- #
# 6. Peptide target with a PTM (phospho-Ser) — both models
# --------------------------------------------------------------------------- #
peptide = [
    "--name", "H2AX", "--target_type", "peptide", "--target_seq", "CKATQASQEY",
    "--binder_id", "A", "--modifications", "SEP", "--modifications_wt", "S",
    "--modifications_positions", "7", "--modification_target", "B",
    "--design_samples", "1", "--gpu_id", "0",
]
run(*peptide, model="boltz2")
run(*peptide, model="boltz1")

# --------------------------------------------------------------------------- #
# 7. Motif scaffolding (template based; boltz2 only)
# --------------------------------------------------------------------------- #
run("--name", "8vc8", "--target_type", "small_molecule", "--target_seq", "HEM",
    "--pdb_motif_id", "A", "--pdb_path", "8vc8", "--use_template", "True",
    "--motif_scaffolding", "True", "--length_min", "140", "--length_max", "150",
    "--motifs", '[{"start_pos": 30, "end_pos": 47}, {"start_pos": 81, "end_pos": 173}]',
    "--min_motif_gap", "15", "--design_samples", "1", "--gpu_id", "0",
    model="boltz2", suffix="boltz2_motif_scaffolding")
