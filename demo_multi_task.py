"""
Multi-Task Demo: Movie Selector Agent
Shows the agent working on multiple different scenarios.
Great for team presentations!
"""

import json
import os
from movie_selector_agent import (
    MovieSelectionTask,
    get_available_movies,
    grade_movie_selection,
)
import openai


DEMO_TASKS = [
    MovieSelectionTask(
        group_size=3,
        preferred_genres=["action", "comedy"],
        max_duration=120,
        expected_movie_title="Action Blast",
    ),
    MovieSelectionTask(
        group_size=2,
        preferred_genres=["romance"],
        max_duration=120,
        expected_movie_title="Romantic Escape",
    ),
    MovieSelectionTask(
        group_size=5,
        preferred_genres=["sci-fi"],
        max_duration=160,
        expected_movie_title="Space Odyssey",
    ),
]


def run_single_agent(client: openai.OpenAI, task: MovieSelectionTask, task_num: int) -> float:
    """Run the agent on a single task and return the reward."""
    
    prompt_text = f"""You are a helpful movie recommendation agent. Your job is to select the perfect movie for a group.

Group Size: {task.group_size} people
Preferred Genres: {', '.join(task.preferred_genres)}
Maximum Duration: {task.max_duration} minutes

You have access to a tool that can fetch available movies. Use it to find the best movie that matches the group's preferences.

After fetching the movies, analyze them carefully and recommend the SINGLE BEST MOVIE by name.
Consider the group size, preferences, and duration constraints.

Your final response should be ONLY the movie title, nothing else."""
    
    messages = [{"role": "user", "content": prompt_text}]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_available_movies",
                "description": "Get a list of available movies based on genre preferences and duration constraints.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preferred_genres": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of preferred movie genres",
                        },
                        "max_duration": {
                            "type": "integer",
                            "description": "Maximum duration in minutes",
                        },
                    },
                    "required": ["preferred_genres"],
                },
            },
        }
    ]
    
    # First LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    
    # Process tool calls if any
    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_available_movies":
                function_args = json.loads(tool_call.function.arguments)
                function_response = get_available_movies(
                    preferred_genres=function_args.get("preferred_genres"),
                    max_duration=function_args.get("max_duration", task.max_duration),
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
        
        # Second LLM call
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_choice = second_response.choices[0].message.content
    else:
        final_choice = response_message.content
    
    # Grade the selection
    reward = grade_movie_selection(final_choice, task.expected_movie_title)
    
    return reward, final_choice


def demo_multi_task():
    """Run the agent on multiple tasks to showcase versatility."""
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set!")
    
    client = openai.OpenAI()
    
    print("\n" + "=" * 80)
    print("🎬 MOVIE SELECTOR AGENT - MULTI-TASK DEMO")
    print("=" * 80)
    print()
    print("This demo shows the agent handling different movie selection scenarios!")
    print()
    
    rewards = []
    
    for i, task in enumerate(DEMO_TASKS, 1):
        print("-" * 80)
        print(f"📋 TASK {i}/{len(DEMO_TASKS)}")
        print("-" * 80)
        print(f"Group Size: {task.group_size} people")
        print(f"Preferences: {', '.join(task.preferred_genres)}")
        print(f"Max Duration: {task.max_duration} minutes")
        print(f"Expected Pick: {task.expected_movie_title}")
        print()
        
        reward, final_choice = run_single_agent(client, task, i)
        rewards.append(reward)
        
        # Determine emoji based on reward
        if reward == 1.0:
            emoji = "✅"
            status = "Perfect!"
        elif reward >= 0.5:
            emoji = "⚠️"
            status = "Same genre"
        else:
            emoji = "❌"
            status = "Wrong pick"
        
        print(f"Agent's Recommendation: {final_choice}")
        print(f"Reward: {reward:.2f}/1.0  {emoji} {status}")
        print()
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Tasks Completed: {len(DEMO_TASKS)}")
    print(f"Perfect Selections: {sum(1 for r in rewards if r == 1.0)}")
    print(f"Average Reward: {sum(rewards) / len(rewards):.2f}")
    print()
    
    if all(r == 1.0 for r in rewards):
        print("🏆 PERFECT PERFORMANCE! All tasks completed successfully!")
    elif sum(rewards) / len(rewards) >= 0.7:
        print("😊 Great job! Agent performed well on most tasks.")
    else:
        print("🤔 Room for improvement - run training with APO to optimize!")
    
    print()
    print("=" * 80)
    print()
    print("💡 Next Steps:")
    print("   1. Try more diverse tasks by editing DEMO_TASKS")
    print("   2. Run 'python prompt_optimization_training.py' for iterative prompt improvement (evaluate → critique → rewrite → re-evaluate)")
    print("   3. Add more genres or constraints to make it harder")
    print()


if __name__ == "__main__":
    demo_multi_task()
