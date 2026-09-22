#!/usr/bin/env python
"""
ni_sweep_targets.py — does num_intra_contacts 4 -> 6 hurt healthy targets at TODAY's
settings? The old PDL1 grid said yes, but that was distogram-only at lr 0.4, fp32,
semi_greedy 1 -- a configuration that no longer exists.

PDL1 and FAD, ni=4 vs ni=6, distogram-only (the current default mode), bf16, 75/45/5,
config lr/helix, semi_greedy 0, LigandMPNN on, AF3 off.
Compare against BHRF1, where ni=6 rescued a 0/4 failure.
"""
import os, queue, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
import overnight_steps as ovn
from calibrate import PDL1_SEQ

TARGETS = {
    "PDL1": {"target_type": "protein", "pdb_target_ids": "A", "target_seq": PDL1_SEQ,
             "use_msa": "True", "msa_max_seqs": "4096",
             "length_min": "90", "length_max": "120"},
    "FAD":  {"target_type": "small_molecule", "target_seq": "FAD",
             "length_min": "130", "length_max": "180"},
}
jobs = queue.Queue()
for t in TARGETS:
    for ni in ["4", "6"]:
        jobs.put((t, ni))

def worker(gpu):
    while True:
        try: t, ni = jobs.get_nowait()
        except queue.Empty: return
        suffix = f"nisweep_ni{ni}"
        if t == "PDL1": ovn.seed_msa("PDL1", suffix)
        base = {"name": t, "boltz_model_version": "boltz2", "pre_iteration": "0",
                "distogram_only": "True", "num_intra_contacts": ni,
                "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
                "semi_greedy_steps": "0", "design_samples": "4", "num_designs": "2",
                "run_boltz_design": "True", "run_ligandmpnn": "True",
                "run_alphafold": "False", "run_rosetta": "False",
                "gpu_id": str(gpu), "suffix": suffix, "work_dir": HERE}
        base.update(TARGETS[t])
        c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
        for k, v in base.items(): c += [f"--{k}", v]
        env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"
        print(f"[{time.strftime('%T')}] start {t} ni={ni} gpu{gpu}", flush=True)
        t0 = time.time()
        with open(os.path.join(HERE, f"nisweep_{t}_ni{ni}.log"), "w") as lf:
            p = subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        print(f"[{time.strftime('%T')}] end {t} ni={ni} rc={p.returncode} wall={time.time()-t0:.0f}s", flush=True)

ths = [threading.Thread(target=worker, args=(g,)) for g in ("1", "2", "4")]
[t.start() for t in ths]; [t.join() for t in ths]
print("[done] ni sweep on PDL1 + FAD")
