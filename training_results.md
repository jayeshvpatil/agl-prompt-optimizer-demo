# Prompt Optimization Results (Current Run)

Script: `prompt_optimization_training.py`
Mode: mock (heuristic selection) | Dataset: `movies_dataset.json` (14 movies)
Split: Train 7 tasks / Val 3 tasks | Early Stopping Patience: 2

## Summary
Baseline Train Accuracy: 71.4%
Baseline Val Accuracy: 66.7%
Best Val Accuracy: 66.7% (no improvement over baseline)
Early Stopped After: 2 optimization rounds (no val lift)

Net Optimization Gain: +0.0 percentage points (validation)

## Interpretation
No generalization improvement; the initial prompt heuristic already matched plateau performance. The persistent error:
- Midnight Terror → Thrill Chase (genre confusion: horror+thriller vs action/thriller proximity)
Also an extraction failure pattern on Action Blast (Unknown) in training set indicates need for stricter output enforcement (now added via `extract_title`).

## Current Failure Modes
| Expected        | Val Wrong | Pattern |
|-----------------|-----------|---------|
| Midnight Terror | Thrill Chase | Genre overlap + rating proximity |

## Recommendations (Next Changes)
1. Add multi-genre intersection tasks (require BOTH genres, not either) to force filtering logic.
2. Introduce decoy thriller/action titles with near-identical ratings to differentiate by genre.
3. Add year/popularity metadata to dataset and incorporate tie-break beyond rating/duration.
4. Run real (non-mock) rounds with small patience to validate if stagnation is due to mock heuristic.
5. Track per-split confusion entropy to observe if error distribution narrows.

## How to Re-run (Real Mode)
```bash
uv run prompt_optimization_training.py --rounds 10 --val-ratio 0.3 --patience 3
```

Mock quick check:
```bash
uv run prompt_optimization_training.py --rounds 5 --mock
```

Inspect results:
```bash
cat prompt_optimization_results.json | jq '.[] | select(.split=="val") | {round, accuracy: (.accuracy*100)}'
```

---
Generated: Updated summary reflecting train/validation split & early stopping.
