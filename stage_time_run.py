import os, re, subprocess, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
from calibrate import PDL1_SEQ
SRC = os.path.join(HERE, "inputs/protein_PDL1_grid_b1_pi0_lr0.1_ni6_hx-0.3/MSA/PDL1_B_env/msa.npz")

def seed(suffix, name):
    for ch in ("A", "B"):
        d = os.path.join(HERE, f"inputs/protein_{name}_{suffix}/MSA", f"{name}_{ch}_env")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "msa.npz")
        if not os.path.exists(dst): shutil.copy2(SRC, dst)

def run(tag, extra):
    base = {"boltz_model_version": "boltz2", "pre_iteration": "0",
            "soft_iteration": "6", "temp_iteration": "3", "hard_iteration": "1",
            "semi_greedy_steps": "0", "init_seed": "42",
            "design_samples": "2", "num_designs": "1", "run_boltz_design": "True",
            "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": "4", "suffix": tag, "work_dir": HERE}
    base.update(extra)
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"; env["BOLTZDESIGN_TIME_STAGES"] = "1"
    log = os.path.join(HERE, f"stagetime_{tag}.log")
    with open(log, "w") as lf:
        subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    t = open(log).read()
    print(f"\n=== {tag} ===")
    for ln in re.findall(r"\[stage\][^\n]*|\[score\][^\n]*", t): print("  " + ln)
    it = [float(x) for x in re.findall(r"Time for iteration \d+:\s*([\d.]+)", t)]
    print(f"  iterations: {len(it)} totalling {sum(it):.1f}s"
          + (f" ({sum(it)/len(it):.2f}s each)" if it else ""))
    print(f"  designs: {len(re.findall(r'^Best sequence', t, re.M))}, error: {'YES' if 'Traceback' in t else 'no'}")

run("stagetime_sm", {"name": "FAD", "target_type": "small_molecule", "target_seq": "FAD",
                     "length_min": "150", "length_max": "150", "num_intra_contacts": "6",
                     "helix_loss_min": "-0.3", "helix_loss_max": "-0.3"})
seed("stagetime_prot", "PDL1")
run("stagetime_prot", {"name": "PDL1", "target_type": "protein", "pdb_target_ids": "A",
                       "target_seq": PDL1_SEQ, "use_msa": "True", "msa_max_seqs": "4096",
                       "length_min": "100", "length_max": "100"})
