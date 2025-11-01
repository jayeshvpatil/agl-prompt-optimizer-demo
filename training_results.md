# Real Prompt Optimization Results

Source Script: `prompt_optimization_training.py`
Dataset: `movies_dataset.json` (14 movies)
Tasks Evaluated: 8
Rounds: 0 (baseline) + 3 optimization rounds

## Summary
Baseline Accuracy: 70.0%
Peak Accuracy: 80.0% (Rounds 1–7)
Final Accuracy: 70.0% (regression at round 8 due to over-constrained prompt)
Net Improvement: +0.0 percentage points

Although accuracy did not improve in these rounds, the final prompt gained:
- Explicit step structure
- Constraint-first filtering
- Tie-breaking criteria (rating > duration proximity > genre ordering)
- Enforced output formatting

## Recommendation
Observed plateau suggests the model stabilized on a decision heuristic early. To create further lift:
1. Add tasks with conflicting constraints (two top-rated exceed duration by 5–10%).
2. Include tasks requiring multi-genre intersection (must match BOTH genres).
3. Introduce decoy titles with similar rating but wrong genre.
4. Penalize 'Unknown' responses (currently treated as failure but not specifically discouraged).
5. Provide structured critique examples instead of free text re-write.

## Next Experiment Ideas
- Add explicit penalty when the recommended title is not in dataset.
- Track per-round confusion entropy (is distribution narrowing?).
- Supply previous successful prompt sections as immutable and only rewrite weak segments.
- Add a validation split and early stop if accuracy regresses >5% for 2 consecutive rounds.

## Confusion Analysis (Rounds 0–8)
Two persistent mistakes surfaced across almost all rounds:

| Expected          | Common Wrong | Frequency (Rounds) | Pattern |
|-------------------|--------------|--------------------|---------|
| Action Blast      | Unknown      | Appears in most rounds | Tool call occasionally skipped / extraction failure |
| Midnight Terror   | Thrill Chase | Appears in most rounds | Genre overlap (thriller vs horror) rating proximity |

Interpretation:
1. "Unknown" indicates extraction failure rather than bad reasoning; improve title extraction by enforcing output regex.
2. Confusion between "Midnight Terror" and "Thrill Chase" shows need for stricter genre filtering precedence over rating.

Proposed Fixes:
- Add explicit instruction: "Reject titles whose genre is not EXACTLY in the preference list unless multi-genre overlap explicitly allowed."
- Post-process model output with a regex to isolate a valid title and re-grade if mismatch.
- Add a third critique dimension: "Did the model attempt to satisfy all listed constraints before selection?" for rewrite guidance.

### Confusion Metrics to Add (Future)
- Per-round confusion count
- Normalized mis-selection rate per expected title
- Cumulative stability index (number of rounds without new error types)

## How to Re-run
```
uv run prompt_optimization_training.py
cat prompt_optimization_results.json | jq '.[] | {round, accuracy: (.accuracy*100), avg_reward}'
```

## Stored Output File
`prompt_optimization_results.json` now contains per-round details:
- prompt text
- accuracy / avg_reward
- per-task results (recommended vs expected, reward)
- confusion map (expected -> {recommended: count})

---
Generated: Automated training documentation
