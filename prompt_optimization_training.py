"""
Real Prompt Optimization Training
=================================

This script performs genuine iterative prompt optimization:
1. Uses the real agent tool (`get_available_movies`) with external dataset
2. Evaluates baseline prompt across tasks (accuracy + avg reward)
3. Generates critique & improved prompt using model itself
4. Re-evaluates new prompt each round
5. Logs all rounds to `prompt_optimization_results.json`

Run:
    uv run prompt_optimization_training.py
"""

import json
import os
import random
import re
import argparse
import openai
from dataclasses import asdict
from typing import List, Dict, Any, Tuple
from movie_selector_agent import (
    MovieSelectionTask,
    get_available_movies,
    grade_movie_selection,
    MOVIES_DATABASE,
)

"""Harder tasks include duration edges, rating conflicts, ambiguous genre overlaps."""
TRAINING_TASKS: List[MovieSelectionTask] = [
    MovieSelectionTask(3, ["action", "comedy"], 120, "Action Blast"),  # baseline good
    MovieSelectionTask(2, ["drama", "romance"], 150, "Sunset Dreams"),  # high rating vs similar genre
    MovieSelectionTask(4, ["horror", "thriller"], 130, "Midnight Terror"),  # mixed dark genres
    MovieSelectionTask(5, ["comedy"], 105, "Comedy of Errors"),  # two comedies similar rating
    MovieSelectionTask(2, ["sci-fi", "adventure"], 140, "Star Quest"),  # choose mid duration vs max
    MovieSelectionTask(3, ["animation", "family"], 90, "Happy Pixels"),  # low duration requirement
    MovieSelectionTask(2, ["action"], 118, "Action Blast"),  # edge duration vs Action Heroes 130
    MovieSelectionTask(3, ["sci-fi"], 150, "Space Odyssey"),  # choose higher rating over Star Quest
    MovieSelectionTask(2, ["comedy"], 100, "Laugh Track"),  # Laugh Track vs Laugh Riot close ratings
    MovieSelectionTask(3, ["thriller", "action"], 120, "Thrill Chase"),  # ambiguous vs Action Blast
]

# Baseline prompt (intentionally minimal)
BASELINE_PROMPT = (
    "You are a movie recommendation agent. Given a task (group size, genres list, max duration):\n"
    "Steps: 1) Fetch movies with tool. 2) Filter STRICTLY: genre must be included; duration must be <= max. 3) Rank by rating desc then duration proximity to max then original genre order. 4) Return ONLY the exact movie title."
)

RESULTS_FILE = "prompt_optimization_results.json"


def extract_title(raw: str, movies: List[Dict[str, Any]]) -> str:
    """Attempt strict extraction of a single movie title.
    - Exact case-insensitive match against dataset titles.
    - If multiple matches appear, choose highest rated among them.
    - If none found, return 'Unknown'.
    """
    lowered = raw.lower()
    matches = []
    for m in movies:
        title = m["title"]
        if title.lower() in lowered:
            matches.append(m)
    if not matches:
        # Try quoted pattern
        quoted = re.findall(r'"([^\"]+)"', raw)
        for q in quoted:
            for m in movies:
                if m["title"].lower() == q.lower():
                    matches.append(m)
        if not matches:
            return "Unknown"
    # Prefer highest rating if ambiguous
    matches.sort(key=lambda x: (-x.get("rating", 0), x.get("duration", 9999)))
    return matches[0]["title"]


def run_agent(client: openai.OpenAI, prompt: str, task: MovieSelectionTask, mock: bool = False) -> Dict[str, Any]:
    """Execute one agent pass with given prompt and task."""
    # First call - ask model how to proceed
    user_block = (
        f"PROMPT:\n{prompt}\n\nTASK:\nGroup Size: {task.group_size}\nGenres: {', '.join(task.preferred_genres)}\nMax Duration: {task.max_duration} min\n"
    )

    if mock:
        # Deterministic mock: always fetch movies (simulate tool call)
        movies = get_available_movies(task.preferred_genres, task.max_duration)
    else:
        response1 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": user_block}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_available_movies",
                        "description": "Get movies matching genres and max duration.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "preferred_genres": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "max_duration": {"type": "integer"},
                            },
                            "required": ["preferred_genres", "max_duration"],
                        },
                    },
                }
            ],
            tool_choice="auto",
        )

        movies = []
        if response1.choices[0].finish_reason == "tool_calls":
            for tool_call in response1.choices[0].message.tool_calls:
                if tool_call.function.name == "get_available_movies":
                    args = json.loads(tool_call.function.arguments)
                    movies = get_available_movies(
                        preferred_genres=args.get("preferred_genres", task.preferred_genres),
                        max_duration=args.get("max_duration", task.max_duration),
                    )
        else:
            movies = get_available_movies(task.preferred_genres, task.max_duration)

    if mock:
        # Heuristic selection for mock mode
        if not movies:
            final_text = "Unknown"
        else:
            # Rank per baseline rules
            ranked = sorted(
                movies,
                key=lambda m: (-m.get("rating", 0), abs(task.max_duration - m.get("duration", 9999))),
            )
            final_text = ranked[0]["title"]
    else:
        movie_payload = json.dumps(movies, indent=2)
        rec_prompt = (
            f"Available Movies JSON:\n{movie_payload}\n\nSelect BEST movie strictly matching genres and duration. Return ONLY title."  # explicit output constraint
        )
        response2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_block},
                {"role": "assistant", "content": response1.choices[0].message.content or ""},
                {"role": "user", "content": rec_prompt},
            ],
            max_tokens=60,
            temperature=0.3,
        )
        final_text = response2.choices[0].message.content.strip()

    # Extract movie title
    recommended_title = extract_title(final_text, movies)

    reward = grade_movie_selection(recommended_title, task.expected_movie_title)
    correct = reward >= 0.8

    return {
        "task": asdict(task),
        "recommended": recommended_title,
        "expected": task.expected_movie_title,
        "reward": reward,
        "correct": correct,
        "raw_response": final_text[:400],
    }


def evaluate_prompt(client: openai.OpenAI, prompt: str, tasks: List[MovieSelectionTask], mock: bool = False) -> Dict[str, Any]:
    """Run prompt over all tasks and build confusion map (expected->recommended counts)."""
    results = [run_agent(client, prompt, t, mock=mock) for t in tasks]
    accuracy = sum(r["correct"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)

    confusion: Dict[str, Dict[str, int]] = {}
    for r in results:
        exp = r["expected"]
        rec = r["recommended"]
        confusion.setdefault(exp, {})
        confusion[exp][rec] = confusion[exp].get(rec, 0) + 1

    return {
        "prompt": prompt,
        "accuracy": accuracy,
        "avg_reward": avg_reward,
        "results": results,
        "confusion": confusion,
    }


def critique_and_rewrite(client: openai.OpenAI, previous_round: Dict[str, Any], failures: List[Dict[str, Any]], mock: bool) -> str:
    """Generate improved prompt via self-critique."""
    if mock:
        # In mock mode just return previous prompt (no real critique) for determinism
        return previous_round["prompt"]

    examples_failures = failures[:5]
    failures_text = "\n".join(
        f"Expected: {f['expected']}, Got: {f['recommended']} (reward={f['reward']:.2f})" for f in examples_failures
    ) or "(No failures captured)"

    critique = f"""You are optimizing a MOVIE RECOMMENDATION AGENT prompt.

CURRENT PROMPT:
{previous_round['prompt']}

PERFORMANCE:
Accuracy: {previous_round['accuracy']*100:.1f}%
Avg Reward: {previous_round['avg_reward']:.2f}

FAILURES (first few):
{failures_text}

IMPROVE THE PROMPT BY:
1. Adding explicit step list
2. Emphasizing constraint filtering BEFORE selection
3. Adding tie-breaking rule on rating > duration proximity > original genre ordering
4. Enforcing output format EXACTLY: movie title only

Return ONLY the new prompt. No explanation, no markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": critique}],
        max_tokens=400,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def save_results(history: List[Dict[str, Any]]):
    with open(RESULTS_FILE, "w") as f:
        json.dump(history, f, indent=2)


def split_tasks(tasks: List[MovieSelectionTask], val_ratio: float, seed: int) -> Tuple[List[MovieSelectionTask], List[MovieSelectionTask]]:
    random.Random(seed).shuffle(tasks)
    val_count = max(1, int(len(tasks) * val_ratio))
    return tasks[val_count:], tasks[:val_count]


def main():
    parser = argparse.ArgumentParser(description="Iterative prompt optimization")
    parser.add_argument("--rounds", type=int, default=8, help="Max optimization rounds")
    parser.add_argument("--val-ratio", type=float, default=0.3, help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on val accuracy")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for task split")
    parser.add_argument("--mock", action="store_true", help="Run without real OpenAI calls (heuristic only)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.mock:
        raise ValueError("OPENAI_API_KEY not set (required unless --mock)")
    client = openai.OpenAI(api_key=api_key) if not args.mock else None  # type: ignore

    print("=" * 80)
    print("REAL PROMPT OPTIMIZATION TRAINING")
    print("=" * 80)
    print(f"Tasks: {len(TRAINING_TASKS)} | Dataset movies: {len(MOVIES_DATABASE)}\n")

    history: List[Dict[str, Any]] = []

    # Baseline evaluation
    # Task split
    train_tasks, val_tasks = split_tasks(TRAINING_TASKS, args.val_ratio, args.seed)
    print(f"Train tasks: {len(train_tasks)} | Val tasks: {len(val_tasks)}")

    print("Baseline evaluation (train + val)...")
    baseline_train = evaluate_prompt(client, BASELINE_PROMPT, train_tasks, mock=args.mock)
    baseline_val = evaluate_prompt(client, BASELINE_PROMPT, val_tasks, mock=args.mock)
    history.append({"round": 0, "split": "train", **baseline_train})
    history.append({"round": 0, "split": "val", **baseline_val})
    print(
        f"Baseline Train Acc: {baseline_train['accuracy']*100:.1f}% | Val Acc: {baseline_val['accuracy']*100:.1f}%\n"
    )

    best_val_acc = baseline_val["accuracy"]
    best_prompt = BASELINE_PROMPT
    patience_counter = 0
    current_prompt = BASELINE_PROMPT
    current_train_round = baseline_train

    for round_idx in range(1, args.rounds + 1):
        print(f"Round {round_idx} optimization...")
        failures = [r for r in current_train_round["results"] if not r["correct"]]
        improved_prompt = critique_and_rewrite(client, current_train_round, failures, mock=args.mock)
        train_eval = evaluate_prompt(client, improved_prompt, train_tasks, mock=args.mock)
        val_eval = evaluate_prompt(client, improved_prompt, val_tasks, mock=args.mock)
        history.append({"round": round_idx, "split": "train", **train_eval})
        history.append({"round": round_idx, "split": "val", **val_eval})

        train_impr = train_eval["accuracy"] - current_train_round["accuracy"]
        val_impr = val_eval["accuracy"] - best_val_acc
        print(
            f"  Train Acc: {train_eval['accuracy']*100:.1f}% (Δ {train_impr*100:+.1f}%) | Val Acc: {val_eval['accuracy']*100:.1f}% (best {best_val_acc*100:.1f}%)"
        )

        # Mis-selections (val set for generalization signal)
        wrong_pairs = []
        for exp, mapping in val_eval["confusion"].items():
            for rec, count in mapping.items():
                if rec != exp:
                    wrong_pairs.append((exp, rec, count))
        wrong_pairs.sort(key=lambda x: -x[2])
        if wrong_pairs:
            print("  Val mis-selections (top 2):")
            for exp, rec, count in wrong_pairs[:2]:
                print(f"    • Expected '{exp}' got '{rec}' x{count}")
        else:
            print("  No val mis-selections this round.")

        # Early stopping logic
        if val_eval["accuracy"] > best_val_acc + 1e-6:
            best_val_acc = val_eval["accuracy"]
            best_prompt = improved_prompt
            patience_counter = 0
            print("  ✅ New best validation accuracy.")
        else:
            patience_counter += 1
            print(f"  (No val improvement) Patience {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print("  🔴 Early stopping triggered.")
                break

        current_train_round = train_eval
        current_prompt = improved_prompt
        print()

    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc*100:.1f}%")
    print(f"Best Prompt:\n{best_prompt}\n")
    save_results(history)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
