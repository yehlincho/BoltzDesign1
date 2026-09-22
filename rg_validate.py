import os, subprocess, sys, threading, time
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import boltz2_sweep as bs, overnight_steps as ovn
from calibrate import PDL1_SEQ
# Does the (now differentiable) --rg_loss hurt healthy targets?
ARMS={"PDL1":("2",{"target_type":"protein","pdb_target_ids":"A","target_seq":PDL1_SEQ,
                   "use_msa":"True","msa_max_seqs":"4096","length_min":"90","length_max":"120"}),
      "FAD": ("4",{"target_type":"small_molecule","target_seq":"FAD",
                   "length_min":"130","length_max":"180"})}
def run(t):
    gpu,extra=ARMS[t]; suffix="rgval_w05"
    if t=="PDL1": ovn.seed_msa("PDL1",suffix)
    base={"name":t,"boltz_model_version":"boltz2","pre_iteration":"0","distogram_only":"True",
          "num_intra_contacts":"4","rg_loss":"0.5",
          "soft_iteration":"75","temp_iteration":"45","hard_iteration":"5","semi_greedy_steps":"0",
          "design_samples":"4","num_designs":"2","run_boltz_design":"True","run_ligandmpnn":"True",
          "run_alphafold":"False","run_rosetta":"False","gpu_id":gpu,"suffix":suffix,"work_dir":HERE}
    base.update(extra)
    c=[bs.CONDA_PY,"-u",os.path.join(HERE,"boltzdesign.py")]
    for k,v in base.items(): c+=[f"--{k}",v]
    env=bs._env_with_conda(); env["BOLTZDESIGN_SEED"]="42"
    print(f"[{time.strftime('%T')}] start {t} rg_loss=0.5 gpu{gpu}",flush=True)
    t0=time.time()
    with open(os.path.join(HERE,f"rgval_{t}.log"),"w") as lf:
        p=subprocess.run(c,stdout=lf,stderr=subprocess.STDOUT,cwd=HERE,env=env)
    print(f"[{time.strftime('%T')}] end {t} rc={p.returncode} wall={time.time()-t0:.0f}s",flush=True)
ths=[threading.Thread(target=run,args=(t,)) for t in ARMS]
[t.start() for t in ths]; [t.join() for t in ths]
