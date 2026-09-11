# BoltzDesign improvements — what we changed & discovered

A running reference of everything done to make BoltzDesign better: the speed wins, the
core design recipe, the fold-diversity / beta-sheet levers, and what was tested and kept
as-is. **All code additions are env-gated — defaults are unchanged unless opted in.**

Last updated: 2026-09.

---

## TL;DR

- **Biggest quality wins:** `pre_iteration=0` + `learning_rate=0.1` (core recipe), and
  fixed `helix_loss=-3` (β-sheet control / escape the small-molecule 4-helix bias).
- **Biggest speed win:** freeze model weights during design — 1.2–1.4× faster,
  **bit-identical** output.
- **Validated no-change:** the optimizer (SGD-on-logits) and the logit parametrization are
  already the best choices; direct-probability and momentum/Adam did not beat them.

---

## 1. Speed (faster generation)

| lever | how | effect | status |
|---|---|---|---|
| **Freeze weights** | `requires_grad_(False)` on all boltz params after model.train()/eval() (~L297) | **1.21× (boltz2) / 1.44× (boltz1)** per step, **bit-identical** loss (max\|Δloss\|=0), **~24% less mem** (boltz1) | **DEFAULT ON** (`BOLTZDESIGN_FREEZE_WEIGHTS=0` to disable) |
| **TF32 matmul** | `set_float32_matmul_precision` (L120) | ~1.47×/step (BoltzHunter-measured); perturbs rounding ~1e-3, NOT bit-exact but distribution-preserving & deterministic given seed | **DEFAULT ON** (`BOLTZDESIGN_MATMUL_PREC=highest` to disable). Pairformer n≥3 receipt still pending (BoltzHunter showed no quality loss) |
| **Fast-BD schedule** | short soft/temp/hard budget | design *stage* ~6×; full pipeline ~1.26× (AF3-dominated) | tested |

**Why freeze works:** design optimizes only `res_type_logits`; the model weights are never
updated but kept `requires_grad=True`, so `total_loss.backward()` computed+accumulated
weight grads every step that the optimizer discarded. Freezing removes that dead-end work.
Logit gradient is mathematically unchanged → provably lossless (stronger than TF32's
distribution-only claim). Details + paper draft in `speed_optimizations.md`.

**Repro:** `freeze_bench.py` (paired same-seed). Also added `BOLTZDESIGN_SEED` for
reproducible A/B — keep it.

## 2. Core design recipe (better designs)

| lever | value | why |
|---|---|---|
| **`pre_iteration`** | **0** | THE dominant lever — the ligand-masked warm-up seeded elongated helices boltz2 never refolds; removing it fixed the main failure mode. Confirmed all targets, both lr. |
| **`learning_rate`** | **0.1** | AF3-proven best. Higher lr → more compact but lower confidence; lr→0.4 anneal FAILED. |
| **helix-loss sign** | negative = suppress | corrected direction (neg suppresses helix). |
| result | boltz2 opt recipe beats boltz1 | 100% vs 87–96% AF3 success on 4 ligands. |

Tooling: `boltz2_sweep.py`, `boltz_grid.py`, `boltz_config_results.ipynb`.

## 3. Fold diversity (escape the 4-helix-bundle bias)

| lever | how | effect |
|---|---|---|
| **Random helicity** | `helix_loss_min/max` range `[-2,0]` (BindCraft-style per-design sample) | FAD 1→4 folds — diversity comes from *spanning* helix↔beta, not extremity (stronger backfires) |
| **`--omit_aa_types CN`** | omit Cys/Asn | fixes the asparagine funnel → doubles backbone diversity |

Mode collapse is a deep landscape funnel; MC-dropout (`BOLTZDESIGN_MC_DROPOUT`) barely
helped and hurt reliability. Deepest fix would be boltz1→boltz2 warm-start (hook exists at
`input_res_type`, not yet plumbed).

## 4. Beta-sheet control (this session)

| lever | value | notes |
|---|---|---|
| **Fixed `helix_loss`** | `min == max == -3` | makes boltz2 real β-sheets — genuine escape from the SM 4-helix bias. Verified (φ/ψ + contact maps), **zero molten**. |
| pLDDT vs strength | U-shaped | strong (−3) is CONFIDENT (0.9); the **medium** penalty (−0.4..−1.0) is the frustrated danger zone (0.5). Strong is safe. |
| boltz2 vs boltz1 | ~10× stiffer | boltz1 flips to β at −0.2; boltz2 needs ~−3 to match. |

**Calibration — tipping point is target-dependent** (`calibrate.py`,
`collected_success/beta_calibration.png`):

| target | tips at |
|---|---|
| PD-L1 (protein) | **−1** (proteins tip easiest; β-target → strand augmentation) |
| SAM (small mol, easy) | **−2** |
| FAD (small mol, resistant) | **−3** |
| universal safe default | **−3** (β everywhere, 0 molten, pLDDT stays high) |

Tooling: `helix_strength.py`, `beta_max.py`, `calibrate.py`. Structures in
`collected_success/beta_sheets_hm3/`. **Abandoned:** explicit `get_beta_loss` term
(`BOLTZDESIGN_BETA_LOSS`) — backfired (rewarded helix bundles), left env-gated/unused.

## 5. Optimizer & parametrization (tested → keep defaults)

| experiment | result |
|---|---|
| **Optimizer** (`BOLTZDESIGN_OPT` sgd/sgd_mom/adamw) | current **SGD-on-logits wins every metric** (pLDDT 0.97, iptm 0.96). Momentum/Adam converge deeper in the soft stage but blow up at the temp→one-hot discretization; SGD recovers best. |
| **Direct-probability** (`BOLTZDESIGN_PROB_OPT` — mirror descent, dL/dp override) | did NOT beat logit update; higher lr wrecks discretization (pLDDT 0.48). |

Same two-phase story both times: alternatives help the *continuous* phase but are fragile
through the *discretization* that sets final quality — the current defaults are well-matched.
Caveat: n=1 (curves deterministic; final pLDDT/iptm single-sample). Untested idea:
momentum-in-soft → SGD-in-hard hybrid. Tooling: `opt_compare.py`, `prob_compare.py`.

## 6. Metrics / tooling

- Success criterion: **AF3 complex pLDDT > 0.7 AND interface ipAE < 10** (`success_rate.py`).
- Ligand-burial metric (healthy band 0.55–0.90) (`ligand_burial.py`).
- Fold diversity via TMalign clustering (`fold_diversity.py`).

---

## Env flags added (all default to unchanged behavior)

| flag | purpose | verdict |
|---|---|---|
| `BOLTZDESIGN_FREEZE_WEIGHTS=1` | freeze weights | **recommend default-on** (free, exact) |
| `BOLTZDESIGN_MATMUL_PREC=high` | TF32 | speed, minor rounding |
| `BOLTZDESIGN_SEED=N` | reproducibility | keep |
| `BOLTZDESIGN_OPT` / `_MOMENTUM` / `_NESTEROV` | optimizer choice | current SGD best; keep default |
| `BOLTZDESIGN_PROB_OPT=1` | direct-prob update | no gain; keep default |
| `BOLTZDESIGN_BETA_LOSS` | explicit β term | backfired; unused |
| `BOLTZDESIGN_MC_DROPOUT` | MC-dropout diversity | weak; unused |

## Open threads

- Flip freeze-weights to default-on for release.
- n=3 confirmation of optimizer/parametrization quality claims.
- AF3 validation of the β-sheet (−3) designs — do they actually bind? (boltz self-confidence
  can be optimistic).
- boltz1→boltz2 warm-start for the stubborn-diversity targets (e.g. SAM).
