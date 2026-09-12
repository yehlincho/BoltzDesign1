# Why `helix_loss` defaults are model-specific

`--helix_loss_min/max` biases binder secondary structure (more negative = suppress helix
→ push toward β). Defaults differ per model because **boltz1 and boltz2 respond ~10×
differently.** An explicit `--helix_loss_*` value overrides the default.

| model | default `(min, max)` | why |
|---|---|---|
| **boltz1** | `(-0.3, 0.0)` | tips to β at ~−0.2, so this spans helix↔β (diverse, confident) |
| **boltz2** | `(-0.3, -0.05)` | ~10× less sensitive; light suppression keeps it confident-helical (needs ~−3 for β) |

**Why not one shared default:** `(-0.3,0)` spans helix↔β on boltz1 (past its −0.2 tipping
point) but is all-helix on boltz2 (~10× less sensitive), so the same number means different
things per model.

**Override examples (boltz2):** β-sheet binder `--helix_loss_min -3 --helix_loss_max -3`;
max-confidence `--helix_loss_min -0.3 --helix_loss_max -0.3`.

*Sensitivity measured mainly on FAD; the `pre_iteration=0 / lr 0.1` recipe is the
AF3-validated part.*
