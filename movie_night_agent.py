import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Mock tool: returns a list of available movies with genres and ratings
def get_available_movies(preferred_genres=None):
    movies = [
        {"title": "Action Blast", "genre": "action", "rating": 8.1},
        {"title": "Laugh Riot", "genre": "comedy", "rating": 7.8},
        {"title": "Space Odyssey", "genre": "sci-fi", "rating": 8.5},
        {"title": "Romantic Escape", "genre": "romance", "rating": 7.2},
        {"title": "Comedy of Errors", "genre": "comedy", "rating": 8.0},
    ]
    if preferred_genres:
        return [m for m in movies if m["genre"] in preferred_genres]
    return movies

# Grader: simple reward function
def grade_choice(choice, expected_genre):
    return 1.0 if expected_genre in choice.lower() else 0.0

def movie_night_agent(task, prompt):
    client = openai.OpenAI()
    messages = [{"role": "user", "content": prompt.format(**task)}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_available_movies",
                "description": "Get a list of available movies by genre.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "preferred_genres": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Preferred genres for the group."
                        }
                    },
                    "required": []
                }
            }
        }
    ]

    # 1. First LLM call to decide if a tool is needed.
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = getattr(response_message, "tool_calls", None)

    # 2. Check if the LLM wants to use a tool.
    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            if function_name == "get_available_movies":
                function_args = json.loads(tool_call.function.arguments)
                function_response = get_available_movies(
                    preferred_genres=function_args.get("preferred_genres")
                )
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                })
        # 3. Second LLM call with tool output
        second_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_choice = second_response.choices[0].message.content
    else:
        final_choice = response_message.content

    # 4. Grade the final choice
    reward = grade_choice(final_choice, task["expected_genre"])
    return final_choice, reward

if __name__ == "__main__":
    task = {
        "group_size": 3,
        "preferred_genres": ["action", "comedy"],
        "expected_genre": "comedy"
    }
    prompt = (
        "You are a movie night agent. The group size is {group_size}. "
        "Preferred genres: {preferred_genres}. Recommend a movie."
    )
    choice, reward = movie_night_agent(task, prompt)
    print(f"Agent's Recommendation: {choice}\nReward: {reward}")
