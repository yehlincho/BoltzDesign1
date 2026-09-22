import os, subprocess, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
import overnight_steps as ovn
from calibrate import BHRF1_SEQ
# distogram-only, ni=4 (defaults) + differentiable Rg penalty at two weights.
# Baseline = the bhrf1_disto_ni4 run already going (same settings, weight 0).
ARMS = {"rg0.5": ("0.5", "2"), "rg2.0": ("2.0", "4")}
def run(arm):
    w, gpu = ARMS[arm]
    suffix = f"dgramrg_{arm}"
    ovn.seed_msa("BHRF1", suffix)
    base = {"name": "BHRF1", "target_type": "protein", "pdb_target_ids": "A",
            "target_seq": BHRF1_SEQ, "use_msa": "True", "msa_max_seqs": "4096",
            "length_min": "90", "length_max": "120",
            "boltz_model_version": "boltz2", "pre_iteration": "0",
            "distogram_only": "True", "num_intra_contacts": "4",
            "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
            "semi_greedy_steps": "0", "design_samples": "4", "num_designs": "2",
            "run_boltz_design": "True", "run_ligandmpnn": "True",
            "run_alphafold": "False", "run_rosetta": "False",
            "gpu_id": gpu, "suffix": suffix, "work_dir": HERE}
    c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
    for k, v in base.items(): c += [f"--{k}", v]
    env = bs._env_with_conda(); env["BOLTZDESIGN_SEED"] = "42"
    env["BOLTZDESIGN_DGRAM_RG"] = w
    print(f"[{time.strftime('%T')}] start {arm} (dgram_rg weight={w}) gpu{gpu}", flush=True)
    t0=time.time()
    with open(os.path.join(HERE, f"bhrf1_{arm}.log"), "w") as lf:
        p = subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
    print(f"[{time.strftime('%T')}] end {arm} rc={p.returncode} wall={time.time()-t0:.0f}s", flush=True)
ths=[threading.Thread(target=run,args=(a,)) for a in ARMS]
[t.start() for t in ths]; [t.join() for t in ths]
