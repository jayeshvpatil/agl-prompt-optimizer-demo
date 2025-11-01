"""
Movie Night Selector Agent - Basic Agent Logic
This module contains the agent logic without agentlightning optimization.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
import openai
import agentlightning as agl


# Load external movie dataset for realism
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "movies_dataset.json")
try:
    with open(_DATASET_PATH, "r") as f:
        MOVIES_DATABASE = json.load(f)
except FileNotFoundError:
    # Fallback minimal dataset if file missing
    MOVIES_DATABASE = [
        {"title": "Action Blast", "genre": "action", "rating": 8.1, "duration": 120},
        {"title": "Laugh Track", "genre": "comedy", "rating": 8.0, "duration": 100},
    ]
except json.JSONDecodeError:
    raise ValueError("movies_dataset.json is not valid JSON")


@dataclass
class MovieSelectionTask:
    """Represents a movie selection task"""
    group_size: int
    preferred_genres: list[str]
    max_duration: int
    expected_movie_title: str  # Ground truth for grading


def get_available_movies(preferred_genres: Optional[list[str]] = None, max_duration: int = 180) -> list[dict]:
    """
    Mock tool: returns a list of available movies filtered by preferences
    """
    movies = MOVIES_DATABASE
    if preferred_genres:
        movies = [m for m in movies if m["genre"] in preferred_genres]
    movies = [m for m in movies if m["duration"] <= max_duration]
    return movies


def grade_movie_selection(selected_movie: str, expected_movie: str) -> float:
    """Grade the agent's movie selection with richer reward shaping.

    Reward tiers:
    - 1.0 exact title match
    - 0.7 same genre & within 10% duration tolerance
    - 0.4 same genre only
    - 0.1 any valid movie title present in dataset
    - 0.0 otherwise
    """
    selected_lower = selected_movie.lower().strip()
    expected_lower = expected_movie.lower().strip()

    # Exact match
    if selected_lower == expected_lower:
        return 1.0

    selected_movie_obj = next((m for m in MOVIES_DATABASE if m["title"].lower() == selected_lower), None)
    expected_movie_obj = next((m for m in MOVIES_DATABASE if m["title"].lower() == expected_lower), None)

    # Genre & duration tolerance
    if selected_movie_obj and expected_movie_obj:
        if selected_movie_obj["genre"] == expected_movie_obj["genre"]:
            expected_duration = expected_movie_obj["duration"]
            sel_duration = selected_movie_obj["duration"]
            if abs(sel_duration - expected_duration) <= expected_duration * 0.1:  # within 10%
                return 0.7
            return 0.4

    # Any valid known movie
    if selected_movie_obj:
        return 0.1

    return 0.0


@agl.rollout
def movie_selector_agent(task: MovieSelectionTask, prompt_template: agl.PromptTemplate) -> float:
    """
    Movie selector agent with agentlightning instrumentation.
    This function is decorated with @agl.rollout to track execution and enable training.
    """
    client = openai.OpenAI()
    
    # Create the prompt using the template
    prompt_text = prompt_template.render(
        group_size=task.group_size,
        preferred_genres=", ".join(task.preferred_genres),
        max_duration=task.max_duration,
    )
    
    messages = [{"role": "user", "content": prompt_text}]
    
    # Define the tool for the LLM
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
    
    # First LLM call to decide if a tool is needed
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)
    
    # Check if the LLM wants to use a tool
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
        
        # Second LLM call with tool output to get final choice
        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        final_choice = second_response.choices[0].message.content
    else:
        final_choice = response_message.content
    
    # Grade the final choice to get a reward
    reward = grade_movie_selection(final_choice, task.expected_movie_title)
    
    return reward
