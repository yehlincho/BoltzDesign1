import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
from calibrate import PDL1_SEQ
import shutil
SRC = os.path.join(HERE, "inputs/protein_PDL1_grid_b1_pi0_lr0.1_ni6_hx-0.3/MSA/PDL1_B_env/msa.npz")
for ch in ("A", "B"):
    d = os.path.join(HERE, "inputs/protein_PDL1_msav2/MSA", f"PDL1_{ch}_env"); os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, "msa.npz")
    if not os.path.exists(dst): shutil.copy2(SRC, dst)
base = {"name": "PDL1", "target_type": "protein", "pdb_target_ids": "A", "target_seq": PDL1_SEQ,
        "use_msa": "True", "msa_max_seqs": "4096", "length_min": "100", "length_max": "100",
        "boltz_model_version": "boltz2", "pre_iteration": "0", "soft_iteration": "10",
        "temp_iteration": "5", "hard_iteration": "2", "semi_greedy_steps": "0", "init_seed": "42",
        "design_samples": "1", "num_designs": "1", "run_boltz_design": "True",
        "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": "2", "suffix": "msav2", "work_dir": HERE}
c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
for k, v in base.items(): c += [f"--{k}", v]
env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"
with open(os.path.join(HERE, "msav2.log"), "w") as lf:
    subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
t = open(os.path.join(HERE, "msav2.log")).read()
print("[verify] score lines:", re.findall(r"\[score\][^\n]*", t) or "NONE")
print("[verify] autocast   :", re.findall(r"\[autocast\][^\n]*", t)[:1] or "NONE")
print("[verify] error      :", "YES" if "Traceback" in t else "no")
