"""
Quick Demo: Movie Selector Agent
This shows the agent in action with a single run (no training required).
Perfect for quick testing and showcasing to your team!
"""

import json
import os
from movie_selector_agent import (
    MovieSelectionTask,
    get_available_movies,
    grade_movie_selection,
    MOVIES_DATABASE,
)
import openai


def demo_agent_single_run():
    """
    Run a single agent execution to demonstrate how it works.
    """
    # Check API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set!")
    
    client = openai.OpenAI()
    
    print("🎬 Movie Selector Agent Demo")
    print("=" * 70)
    print()
    
    # Create a sample task
    task = MovieSelectionTask(
        group_size=3,
        preferred_genres=["action", "comedy"],
        max_duration=120,
        expected_movie_title="Action Blast",
    )
    
    print("📋 Task Details:")
    print(f"   Group Size: {task.group_size} people")
    print(f"   Preferred Genres: {', '.join(task.preferred_genres)}")
    print(f"   Max Duration: {task.max_duration} minutes")
    print(f"   Expected Selection: {task.expected_movie_title}")
    print()
    print("-" * 70)
    print()
    
    # Step 1: Create the prompt
    prompt_text = f"""You are a helpful movie recommendation agent. Your job is to select the perfect movie for a group.

Group Size: {task.group_size} people
Preferred Genres: {', '.join(task.preferred_genres)}
Maximum Duration: {task.max_duration} minutes

You have access to a tool that can fetch available movies. Use it to find the best movie that matches the group's preferences.

After fetching the movies, analyze them carefully and recommend the SINGLE BEST MOVIE by name.
Consider the group size, preferences, and duration constraints.

Your final response should be ONLY the movie title, nothing else."""
    
    messages = [{"role": "user", "content": prompt_text}]
    
    # Define the tool
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
    
    print("🤖 Step 1: Sending initial request to LLM...")
    print("   (Agent decides whether to use tools)")
    print()
    
    # First LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    
    # Check if tool was called
    if tool_calls:
        print(f"✅ LLM decided to use tool: {tool_calls[0].function.name}")
        print()
        
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_available_movies":
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Step 2: Executing Tool")
                print(f"   Tool: {function_name}")
                print(f"   Parameters: {function_args}")
                print()
                
                # Execute the tool
                function_response = get_available_movies(
                    preferred_genres=function_args.get("preferred_genres"),
                    max_duration=function_args.get("max_duration", task.max_duration),
                )
                
                print(f"📚 Available Movies Found: {len(function_response)}")
                for movie in function_response:
                    print(f"   • {movie['title']} ({movie['genre']}, {movie['duration']}min, ⭐{movie['rating']})")
                print()
                
                # Add tool response to messages
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
        
        print("🤖 Step 3: Sending tool results back to LLM...")
        print("   (Agent decides on final recommendation)")
        print()
        
        # Second LLM call
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_choice = second_response.choices[0].message.content
    else:
        print("⚠️  LLM decided not to use tools")
        final_choice = response_message.content
    
    print(f"🎬 Final Recommendation: {final_choice}")
    print()
    
    # Grade the selection
    reward = grade_movie_selection(final_choice, task.expected_movie_title)
    
    print("-" * 70)
    print()
    print(f"📊 Evaluation:")
    print(f"   Expected: {task.expected_movie_title}")
    print(f"   Recommended: {final_choice}")
    print(f"   Reward: {reward:.2f}/1.0")
    
    if reward == 1.0:
        print("   ✅ Perfect selection!")
    elif reward == 0.5:
        print("   ⚠️  Same genre but different movie")
    else:
        print("   ❌ Incorrect selection")
    
    print()
    print("=" * 70)
    print()
    print("🎉 Demo Complete!")
    print()
    print("Next Steps:")
    print("   • Run 'python prompt_optimization_training.py' to see real iterative prompt refinement.")
    print("   • That script evaluates the current prompt across harder tasks, critiques failures, rewrites, and re-evaluates.")
    print("   • Results (accuracy, reward trend, confusion analysis) are saved to prompt_optimization_results.json and training_results.md.")


if __name__ == "__main__":
    demo_agent_single_run()
