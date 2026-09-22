import os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import boltz2_sweep as bs
# BHRF1 from a template (7P33 chain A: BHRF1 with Bid BH3 in the groove, i.e. the
# binding-competent conformation) instead of a 12-sequence MSA. Same arm settings as
# the cth50 MSA run so only the target conditioning differs.
base = {"name": "BHRF1tmpl", "pdb_path": "7P33", "target_type": "protein", "pdb_target_ids": "A",
        "use_template": "True", "use_msa": "False",
        "length_min": "90", "length_max": "120",
        "boltz_model_version": "boltz2", "pre_iteration": "0",
        "distogram_only": "False",
        "soft_iteration": "75", "temp_iteration": "45", "hard_iteration": "5",
        "semi_greedy_steps": "0", "design_samples": "4", "num_designs": "2",
        "run_boltz_design": "True", "run_ligandmpnn": "True",
        "run_alphafold": "False", "run_rosetta": "False",
        "gpu_id": "1", "suffix": "tmpl_cth50", "work_dir": HERE}
c = [bs.CONDA_PY, "-u", os.path.join(HERE, "boltzdesign.py")]
for k, v in base.items(): c += [f"--{k}", v]
env = bs._env_with_conda()
env["BOLTZDESIGN_SEED"] = "42"
env["BOLTZDESIGN_DESIGN_SAMPLING_STEPS"] = "50"
env["BOLTZDESIGN_CONFIDENCE_FROM"] = "temp"
print("[tmpl] launching:", " ".join(c[-8:]), flush=True)
t0 = time.time()
with open(os.path.join(HERE, "bhrf1_template.log"), "w") as lf:
    p = subprocess.run(c, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE, env=env)
print(f"[tmpl] rc={p.returncode} wall={time.time()-t0:.0f}s")
