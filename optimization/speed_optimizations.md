# BoltzDesign speed optimizations — where they live & what's portable

Reference for the model/pipeline speed work across the three codebases
(**Pairformer**, **BoltzHunter**, **AlphaFold3 fork**), what was measured, and
what is (not) active in this repo (Pairformer).

**TL;DR** — the model-level speedups were built and shipped in **BoltzHunter**
(~1.4× net end-to-end, no quality loss). **Pairformer runs the un-optimized path.**
The biggest win (TF32) is a one-line port.

**NEW (Pairformer-native, 2026-09): freeze model weights during design — 1.2–1.4× per
step, BIT-IDENTICAL output.** Strictly stronger claim than TF32 (which perturbs rounding):
freezing changes *nothing* numerically. See the dedicated section below. This is the
first speedup that is both Pairformer-native and provably zero-cost.

---

## Freeze model weights during the design loop (Pairformer-native, bit-identical)

**What.** BoltzDesign optimizes only `res_type_logits`; the boltz model weights are never
updated. But they keep `requires_grad=True`, so `total_loss.backward()`
(`boltzdesign_utils.py`) computes **and accumulates** a gradient for every model weight
each step — which the optimizer then ignores (`optimizer.zero_grad()` only clears the
logits). Pure wasted compute + memory. Freezing the weights skips the weight-gradient
matmuls and drops the activations only kept for them.

**How (env-gated, default off):** `BOLTZDESIGN_FREEZE_WEIGHTS=1`. After the
`model.train()/eval()` line (~L297): `for p in boltz_model.parameters(): p.requires_grad_(False)`.
The logit gradient `dL/d(logits)` is mathematically unchanged (weights act as constants),
so the optimization is numerically identical.

**Benchmarked** (`freeze_bench.py`, paired same-seed, FAD length-150, A100; correctness =
full-precision `total_loss` trajectory over 47 steps):

| model | correctness | speed (s/iter) | speedup | peak GPU mem |
|---|---|---|---|---|
| **boltz2** | **max\|Δloss\| = 0** (bit-identical) | 2.81 → 2.32 | **1.21×** | drop |
| **boltz1** | **max\|Δloss\| = 0** (bit-identical) | 3.16 → 2.20 | **1.44×** | **~12.4 → 9.4 GB (~24%)** |

**Why it matters:** unlike TF32, freezing is provably lossless (identical trajectory, not
just identical distribution), so it needs no n≥3 distributional caveat. Stacks additively
with TF32 (independent mechanisms: TF32 = matmul precision, freeze = fewer backward matmuls).
Recommend **default-on**. Also added `BOLTZDESIGN_SEED` (seeds random/np/torch/cuda) for
reproducible A/B — keep it.

---

## Benchmarked results (BoltzHunter, 2026-08-11)

Source: `BoltzHunter/runs/speedtest/tf32_speedup_report.md`
· memory: `speedups-tf32-batched-eval`.

| optimization | speedup | quality | status |
|---|---|---|---|
| **TF32 matmul** — `set_float32_matmul_precision("high")` | **1.47× / design step** (4.478 → 3.045 s) | zero loss (pLDDT 0.592 = 0.592) | **shipped, default in BoltzHunter** |
| **Batched eval** — one `boltz2 predict` load, holo + apo merged | **1.83× / ligand** (~262 → 143 s) | unchanged | shipped |
| **trifast** — fused triangle attention (`BOLTZHUNTER_TRIFAST=1`) | ~9% **slower** @150-aa/empty-MSA | — | **tested & rejected** (default off) |
| **Net (design + eval run)** | **~1.4× wall-clock** | free | — |

Notes:
- TF32's per-step 1.47× scales with design length (more steps → more saving); wall
  speedup is lower only because the fixed ~130 s model-load isn't matmul-bound.
- Batched eval wins because eval was >50% cold model-load overhead — merging holo+apo
  into one prediction dir removes the second load.
- trifast was rejected because triangle-attention is not the bottleneck at 150-aa /
  empty-MSA; re-test only for long or MSA-backed targets.
- **Improved numerical scoring** in the original wishlist ≈ the TF32 precision change.

---

## Where each optimization lives

| optimization | Pairformer (this repo) | BoltzHunter | AlphaFold3 fork (`~/alphafold3`) |
|---|---|---|---|
| TF32 matmul | ❌ slow `"highest"` (FP32) | ✅ `"high"` default | n/a (JAX) |
| Batched holo+apo eval | ❌ | ✅ | n/a |
| apo-skip (`--apo_every N`) | ❌ apo every step | ✅ | n/a |
| Featurization cache (`_FEAT_CACHE`) | ❌ | ✅ | n/a |
| trifast / fused kernels | ❌ (`use_trifast=False`) | tested, off | n/a |
| PyRosetta optional | ✅ `--run_rosetta` (default off for SM) | ✅ (post-filter optional) | n/a |
| JAX kernel suite (flash, cuequivariance, warm compile cache, async compile) | n/a | n/a | ✅ (JAX/AF3 only) |

**boltz2 fast kernels** (`use_kernels`, `use_cuequiv_mul/attn`, `use_trifast`) exist in
the vendored `boltz2/src/boltz/model/layers/pairformer.py` but are **off in Pairformer's
design path** — they are inference/forward-only fused kernels that don't support the
gradient backprop the design loop needs, which is why `use_trifast=False` is hardcoded.

---

## What Pairformer currently does (the slow path)

- `boltzdesign/boltzdesign_utils.py:116` → `torch.set_float32_matmul_precision("highest")`
  (full FP32; also in 9 sibling `boltzdesign_*` files). **← the 1.47× is left on the table here.**
- `boltzdesign/boltzdesign_utils.py:1374` and `:1465` → `_run_model(..., best_batch_apo, ...)`
  runs the **apo prediction every step** (BoltzHunter's `--apo_every` skips this).
- No featurization cache; eval not batched.

---

## Porting to Pairformer (recommended, in priority order)

1. **TF32 (one line, ~1.47×/step, zero risk).**
   `boltzdesign_utils.py:116`: `"highest"` → `"high"`.
   Optionally env-gate it like BoltzHunter:
   ```python
   torch.set_float32_matmul_precision(os.environ.get("BOLTZDESIGN_MATMUL_PREC", "high"))
   ```
   Report shows mean pLDDT unchanged and backward pass fine. **Do this first.**

2. **apo-skip `--apo_every N` (~1.6× on small molecule).**
   Add a CLI flag in `boltzdesign.py`; in `boltzdesign_utils.py` guard the apo
   `_run_model` calls (`:1374`, `:1465`) to run only every Nth step and reuse the last
   apo output in between. Safe for small molecule (RMSD not in the SM filter); for
   protein keep apo≈every-3 (RMSD is in the filter).

3. **Fast kernels on forward-only calls (optional).**
   The final re-prediction and apo prediction are pure inference (no grad) — could pass
   `use_kernels=True` there without touching the differentiable design path.

4. **Batched eval (~1.83×/ligand).**
   Larger change: merge holo+apo into one `boltz2 predict` dir at the eval stage.

**Validation:** benchmark on a FAD length-150 design (paired, same seeds, same GPU),
compare per-step seconds and mean pLDDT before/after — the report's protocol.

---

## For the bioRxiv paper (draft text)

> **Computational efficiency.** Report these as acceleration methods in the Methods /
> Supplementary, with the paired-benchmark numbers above.

**Methods (draft):** "To accelerate gradient-based design we enabled TF32 tensor-core
matmuls on the design and backward passes (`torch.set_float32_matmul_precision('high')`),
which reduced per-step cost 1.47× (4.48 → 3.05 s/step on an A100, FAD length-150 binder)
with no change in mean design-model pLDDT (0.592 vs 0.592; paired, identical seeds). At
evaluation we merged the holo and apo co-folds into a single model-load, giving a 1.83×
per-ligand speedup (eval was >50% cold model-load overhead). Fused triangle-attention
(trifast) kernels were tested but gave no benefit at the ~150-residue, single-sequence
scale of these designs (~9% slower) and were left disabled. Together these give ~1.4×
end-to-end wall-clock at no measurable cost to design quality, which is gated downstream
by AlphaFold3 co-fold validation."

**Freeze-weights addition (draft):** "Because the design procedure optimizes only the
input sequence logits and never the network parameters, we disabled gradient tracking on
all model weights during the design loop. This removes the weight-gradient computation and
its activation storage from every backward pass while leaving the logit gradient exactly
unchanged; per-step cost fell 1.21× (boltz2) and 1.44× (boltz1) with a bit-identical
optimization trajectory (max|Δloss| = 0 over 47 steps, paired identical seeds) and ~24%
lower peak GPU memory (boltz1). Unlike the TF32 change this speedup is numerically exact."

**Reproducibility to state:** paired A/B, same GPU, sequential, identical seeds/masked
inputs; quality gated by AF3 (ipTM / pLDDT / interface ipAE), which is independent of
design-loop matmul precision. Freeze-weights is exact (identical trajectory); TF32 is
distribution-preserving only.

**Figure suggestion:** bar chart of per-step time (FP32 vs TF32) + per-ligand eval time
(2-load vs 1-load), with a paired-pLDDT inset showing no quality change. Data:
`BoltzHunter/runs/speedtest/{FAD_highest.log, FAD_high.log}`.

**Caveats to disclose:** TF32 perturbs matmul rounding (~1e-3), so individual design
trajectories diverge from FP32 even at equal seeds; only the *distribution* of outcomes
is preserved (n=3 in the pilot — widen before publication). Speedups measured in
BoltzHunter; Pairformer must port TF32 (§Porting) to obtain them.
