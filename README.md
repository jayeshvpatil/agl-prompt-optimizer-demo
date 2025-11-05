# Movie Selector Agent – Lean Demo & Prompt Optimization

This repository contains a minimal, runnable AI agent that recommends a movie using tool calls and an iterative prompt optimization loop.

## Contents You Actually Need

| File | Purpose |
|------|---------|
| `demo.py` | Single-task demo (quick showcase) |
| `demo_multi_task.py` | Multiple scenarios summary |
| `movie_selector_agent.py` | Core agent logic (tool + grading) |
| `prompt_optimization_training.py` | Iterative prompt improvement (evaluate → critique → rewrite → re-evaluate) |
| `movies_dataset.json` | Realistic movie metadata source |
| `training_results.md` | Latest optimization run summary |

Everything else is optional. You can delete the other markdown files if you want a leaner repo.

## Quick Start

```bash
# 1. Ensure Python 3.12+ and uv installed
python --version
pip install uv  # if not installed

# 2. Add your API key
echo "OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE" > .env

# 3. Install dependencies
uv sync

# 4. Run the quick demo
uv run demo.py

# 5. (Optional) Run multi-task demo
uv run demo_multi_task.py

# 6. Run iterative prompt optimization
uv run prompt_optimization_training.py
```

## What You’ll See

Demo output (example):
```
🎬 Final Recommendation: Action Blast
📊 Evaluation: Expected=Action Blast, Reward=1.00 ✅ Perfect selection
```

Optimization script output (truncated):
```
Baseline Accuracy: 70.0%
Round 1 Accuracy: 80.0% (+10.0%)
Rounds 2–7: 80.0%
Round 8: 70.0% (regression)
Persistent misses: Action Blast → Unknown; Midnight Terror → Thrill Chase
```
Artifacts saved to:
* `prompt_optimization_results.json`
* `training_results.md`

## How Prompt Optimization Works

1. Evaluate current prompt on a set of harder tasks
2. Collect failures & confusion map
3. Critique prompt (LLM explains weaknesses)
4. Rewrite prompt incorporating feedback
5. Re-evaluate and track accuracy & reward trend
6. Repeat for N rounds (defaults to 8)

## Minimal Customization

| Task | Where to Change |
|------|-----------------|
| Add movies | `movies_dataset.json` |
| Adjust reward logic | `grade_movie_selection()` in `movie_selector_agent.py` |
| Harder tasks | Edit task list in `prompt_optimization_training.py` |
| Reduce cost | Lower rounds or switch to `gpt-3.5-turbo` |


## License & Notes

Internal demo code. Adapt pattern for routing, recommendations, extraction, or evaluation agents.

---
Enjoy iterating! Keep runs short and focused for lower cost.

