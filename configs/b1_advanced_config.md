# B1 Advanced Config

This document describes the implemented config surface for the enhanced B1 line.

## Naming Rule

- Knobs that affect Phase 1 outputs, OOF cache generation, or sample weighting live under `training`
- Knobs that only change the final Phase 2 decision stay under `stacking`
- B1-specific knobs are grouped under `training.ml_hierarchical`

## Implemented Layout

```json
{
  "training": {
    "ml_hierarchical_training": true,
    "ml_hierarchical": {
      "stage1": {
        "resampling": {
          "strategy": "smote",
          "target_ratio": 0.9,
          "k_neighbors": 5
        },
        "weighting": {
          "mode": "effective_number",
          "beta": 0.999
        }
      },
      "stage2": {
        "resampling": {
          "strategy": "none",
          "target_ratio": 1.0,
          "k_neighbors": 5
        },
        "weighting": {
          "mode": "none",
          "beta": 0.999
        }
      },
      "stage_features": {
        "enabled": true,
        "include_positive_prob": true
      }
    }
  },
  "stacking": {
    "meta_features": "proba+uncertainty+stage"
  }
}
```

## B1.1: Separate Resampling Per Stage

Purpose: allow Stage 1 and Stage 2 to use different rebalancing policies.

Implemented fields:

- `training.ml_hierarchical.stage1.resampling.strategy`
- `training.ml_hierarchical.stage1.resampling.target_ratio`
- `training.ml_hierarchical.stage1.resampling.k_neighbors`
- `training.ml_hierarchical.stage2.resampling.strategy`
- `training.ml_hierarchical.stage2.resampling.target_ratio`
- `training.ml_hierarchical.stage2.resampling.k_neighbors`

Implemented strategies:

- `none`
- `smote`
- `random_over`

Notes:

- `target_ratio = 0.9` means oversample minority classes to 90% of the majority count
- `stage2` defaults to `none` because the clay/sand split is usually close to balanced

## B1.2: Explicit Stage Uncertainty Features

Purpose: keep Stage 1 / Stage 2 binary confidence visible to the Phase 2 meta learner.

Implemented fields:

- `training.ml_hierarchical.stage_features.enabled`
- `training.ml_hierarchical.stage_features.include_positive_prob`
- `stacking.meta_features = proba+uncertainty+stage`

When enabled, each GB/XGB branch exports:

- Stage 1 positive probability (`P(silt)`) when `include_positive_prob = true`
- Stage 1 uncertainty summary: `max_prob`, `margin`, `entropy`
- Stage 2 positive probability when `include_positive_prob = true`
- Stage 2 uncertainty summary: `max_prob`, `margin`, `entropy`

These arrays are stored in the OOF cache as:

- `oof_gb_stage_meta`
- `oof_xgb_stage_meta`

## B1.3: Class-Balanced Weighting

Purpose: avoid relying on oversampling alone by injecting class-aware sample weights into Stage 1 / Stage 2 fitting.

Implemented fields:

- `training.ml_hierarchical.stage1.weighting.mode`
- `training.ml_hierarchical.stage1.weighting.beta`
- `training.ml_hierarchical.stage2.weighting.mode`
- `training.ml_hierarchical.stage2.weighting.beta`

Implemented modes:

- `none`
- `inverse_freq`
- `effective_number`

Notes:

- weights are computed from the original stage labels
- the resulting class weights are then materialized onto the resampled dataset before `fit(...)`

## Practical Recommendation

- If you change `training.ml_hierarchical.*`, treat it as a Phase 1 change and regenerate OOF cache
- If you only change `stacking.stage1_decision` / `stacking.silt_threshold`, Phase 2-only remains valid

Recommended overlays:

- `config.experiment_b1.json`: current enhanced B1 default
- optional future overlays:
  - `config.experiment_b1_stage1_aggressive.json`
  - `config.experiment_b1_stage2_random_over.json`
  - `config.experiment_b1_inverse_freq.json`
