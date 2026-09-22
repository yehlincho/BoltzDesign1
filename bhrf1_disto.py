#!/usr/bin/env python
"""
bhrf1_disto.py — completes the 2x2 on BHRF1 at TODAY's settings:
              ni=4                     ni=6
  confidence  ovn_cth50 (0/4 <2A)      nitest_ni6 (2/2 <2A so far)
  distogram   <- this run              <- this run

Same everything else as the confidence arms (bf16, 75/45/5, lr/helix from config,
semi_greedy 0, LigandMPNN on, AF3 off). Sequential on one GPU; distogram-only is fast.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
import overnight_steps as ovn
from calibrate import BHRF1_SEQ

for ni in ["4", "6"]:
    suffix = f"disto_ni{ni}"
    ovn.seed_msa("BHRF1", suffix)
    base = {"name": "BHRF1", "target_type": "protein", "pdb_target_ids": "A",
            "target_seq": BHRF1_SEQ, "use_msa": "True", "msa_max_seqs": "4096",
            "length_min": "90", "length_max": "120",
            "boltz_model_version": "boltz2", "pre_iteration": "0",
            "distogram_only": "True", "num_intra_contacts": ni,
            "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
            "semi_greedy_steps": "0", "design_samples": "4", "num_designs": "2",
            "run_boltz_design": "True", "run_ligandmpnn": "True",
            "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": "0", "suffix": suffix, "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"
    print(f"[{time.strftime('%T')}] start distogram-only ni={ni}", flush=True)
    t0 = time.time()
    with open(os.path.join(HERE, f"bhrf1_disto_ni{ni}.log"), "w") as lf:
        p = subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    print(f"[{time.strftime('%T')}] end ni={ni} rc={p.returncode} wall={time.time()-t0:.0f}s", flush=True)
