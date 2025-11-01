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
import openai
from dataclasses import asdict
from typing import List, Dict, Any
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
    "Recommend the best movie for this group. Use the tool to fetch movies, then pick the single best title."  # intentionally vague
)

RESULTS_FILE = "prompt_optimization_results.json"


def run_agent(client: openai.OpenAI, prompt: str, task: MovieSelectionTask) -> Dict[str, Any]:
    """Execute one agent pass with given prompt and task."""
    # First call - ask model how to proceed
    user_block = (
        f"PROMPT:\n{prompt}\n\nTASK:\nGroup Size: {task.group_size}\nGenres: {', '.join(task.preferred_genres)}\nMax Duration: {task.max_duration} min\n"
    )

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
        # Model skipped tool - still supply filtered movies for fairness
        movies = get_available_movies(task.preferred_genres, task.max_duration)

    # Second call - recommendation
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
        max_tokens=100,
        temperature=0.4,
    )

    final_text = response2.choices[0].message.content.strip()

    # Extract movie title
    recommended_title = "Unknown"
    for m in movies:
        if m["title"].lower() in final_text.lower():
            recommended_title = m["title"]
            break

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


def evaluate_prompt(client: openai.OpenAI, prompt: str) -> Dict[str, Any]:
    """Run prompt over all tasks and build confusion map (expected->recommended counts)."""
    results = [run_agent(client, prompt, t) for t in TRAINING_TASKS]
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


def critique_and_rewrite(client: openai.OpenAI, previous_round: Dict[str, Any]) -> str:
    """Generate improved prompt via self-critique."""
    examples_failures = [r for r in previous_round["results"] if not r["correct"]][:3]
    failures_text = "\n".join(
        f"Expected: {f['expected']}, Got: {f['recommended']} (reward={f['reward']})" for f in examples_failures
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


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = openai.OpenAI(api_key=api_key)

    print("=" * 80)
    print("REAL PROMPT OPTIMIZATION TRAINING")
    print("=" * 80)
    print(f"Tasks: {len(TRAINING_TASKS)} | Dataset movies: {len(MOVIES_DATABASE)}\n")

    history: List[Dict[str, Any]] = []

    # Baseline evaluation
    print("Baseline evaluation...")
    baseline_round = evaluate_prompt(client, BASELINE_PROMPT)
    history.append({"round": 0, **baseline_round})
    print(f"Baseline Accuracy: {baseline_round['accuracy']*100:.1f}% | Avg Reward: {baseline_round['avg_reward']:.2f}\n")

    current_round = baseline_round
    current_prompt = BASELINE_PROMPT

    total_rounds = 8
    # Iterative optimization rounds (extended)
    for round_idx in range(1, total_rounds + 1):
        print(f"Round {round_idx} optimization...")
        improved_prompt = critique_and_rewrite(client, current_round)
        eval_round = evaluate_prompt(client, improved_prompt)
        history.append({"round": round_idx, **eval_round})

        improvement = eval_round["accuracy"] - current_round["accuracy"]
        print(
            f"  Accuracy: {eval_round['accuracy']*100:.1f}% (Δ {improvement*100:+.1f}%) | Avg Reward: {eval_round['avg_reward']:.2f}"
        )

        # Print top 2 confusion issues if any
        wrong_pairs = []
        for exp, mapping in eval_round["confusion"].items():
            for rec, count in mapping.items():
                if rec != exp:
                    wrong_pairs.append((exp, rec, count))
        wrong_pairs.sort(key=lambda x: -x[2])
        if wrong_pairs:
            print("  Most common mis-selections:")
            for exp, rec, count in wrong_pairs[:2]:
                print(f"    • Expected '{exp}' got '{rec}' x{count}")
        else:
            print("  No mis-selections this round.")

        current_round = eval_round
        current_prompt = improved_prompt
        print()

    # Final summary
    print("=" * 80)
    final = history[-1]
    baseline = history[0]
    total_improvement = final["accuracy"] - baseline["accuracy"]
    print(
        f"Final Accuracy: {final['accuracy']*100:.1f}% (Baseline {baseline['accuracy']*100:.1f}%, Δ {total_improvement*100:+.1f}%)"
    )
    print(f"Final Prompt:\n{current_prompt}\n")

    save_results(history)
    print(f"Results saved to {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
