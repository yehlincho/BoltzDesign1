#!/usr/bin/env python
"""
msa_score_bench.py — verify full-MSA final prediction on a PROTEIN target.

  full_msa  : default now (scoring uses every MSA row)
  subsampled: BOLTZDESIGN_SCORE_SUBSAMPLE=1 (old behaviour, 1024 random rows)

--init_seed pinned so both arms score an identical design. Sequential, one GPU.
Also runs the full_msa arm twice to check score reproducibility, which subsampling
previously broke (fresh randperm every call, not gated on training).
"""
import os, re, subprocess, sys, time, argparse, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import boltz2_sweep as bs
from calibrate import PDL1_SEQ
MSA_SRC = os.path.join(HERE, "inputs/protein_PDL1_grid_b1_pi0_lr0.1_ni6_hx-0.3/MSA/PDL1_B_env/msa.npz")
ARMS = {"full_msa": {}, "full_msa_rep2": {}, "subsampled": {"BOLTZDESIGN_SCORE_SUBSAMPLE": "1"}}

def seed_msa(suffix):
    main = os.path.join(HERE, f"inputs/protein_PDL1_{suffix}")
    for ch in ("A", "B"):
        d = os.path.join(main, "MSA", f"PDL1_{ch}_env"); os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "msa.npz")
        if not os.path.exists(dst): shutil.copy2(MSA_SRC, dst)

def cmd(gpu, suffix):
    base = {"name": "PDL1", "target_type": "protein", "pdb_target_ids": "A",
            "target_seq": PDL1_SEQ, "use_msa": "True", "msa_max_seqs": "4096",
            "length_min": "100", "length_max": "100", "boltz_model_version": "boltz2",
            "pre_iteration": "0", "soft_iteration": "20", "temp_iteration": "10",
            "hard_iteration": "2", "semi_greedy_steps": "0", "init_seed": "42",
            "design_samples": "1", "num_designs": "1", "run_boltz_design": "True",
            "run_ligandmpnn": "False", "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    return c

def run(gpu, arm):
    suffix = f"msascore_{arm}"; seed_msa(suffix)
    env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"; env.update(ARMS[arm])
    log = os.path.join(HERE, f"msa_score_{arm}.log")
    with open(log, "w") as lf:
        subprocess.run(cmd(gpu, suffix), stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    t = open(log).read()
    return {"score_s": [float(x) for x in re.findall(r"\[score\] predict_step took ([\d.]+)s", t)],
            "full_tag": t.count("(full MSA)"),
            "holo": re.findall(r"^Holo Complex PLDDT:\s*([\d.]+)", t, re.M),
            "apo": re.findall(r"^Apo Complex PLDDT:\s*([\d.]+)", t, re.M),
            "rmsd": re.findall(r"^RMSD:\s*([\d.]+)", t, re.M),
            "seq": (re.findall(r"^Best sequence:\s*([A-Z]+)", t, re.M) or [None])[0]}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gpu", default="1"); a = ap.parse_args()
    P = {}
    for arm in ARMS:
        print(f"[{time.strftime('%T')}] {arm}...", flush=True); P[arm] = run(a.gpu, arm)
    L = ["========= FULL-MSA SCORING ON A PROTEIN TARGET (PDL1) =========",
         "binder 100aa | msa_max_seqs 4096 | init_seed 42 | 32-iter schedule", ""]
    for arm in ARMS:
        d = P[arm]
        L.append(f"{arm:14} score={sum(d['score_s']):6.2f}s {d['score_s']} "
                 f"'(full MSA)' tags={d['full_tag']} holo={d['holo']} apo={d['apo']} rmsd={d['rmsd']}")
    same = P["full_msa"]["seq"] == P["subsampled"]["seq"] == P["full_msa_rep2"]["seq"]
    L += ["", f"same design in all arms: {'YES' if same else 'NO'}",
          "reproducibility of full-MSA scoring (rep1 vs rep2): "
          f"holo {P['full_msa']['holo']} vs {P['full_msa_rep2']['holo']}", "=" * 62]
    rep = "\n".join(L); print(rep)
    open(os.path.join(HERE, "msa_score_report.txt"), "w").write(rep + "\n")

if __name__ == "__main__":
    main()
