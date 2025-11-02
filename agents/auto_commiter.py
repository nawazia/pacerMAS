#!/usr/bin/env python

# This script implements a multi-agent system using LangGraph to monitor
# a local Git repository and automatically commit changes related to a
# specific feature.
#
# Requirements:
# pip install langchain langgraph langchain-google-genai gitpython python-dotenv
#
# Setup:
# 1. Make sure you have a .env file in the same directory with your
#    Google Gemini API key:
#    GOOGLE_API_KEY="your_api_key_here"
#
# 2. Create a feature file (e.g., feature.txt) with your feature
#    description and acceptance criteria.
#
# 3. Ensure the target repository is a valid Git repo and you have
#    push permissions to the 'origin' remote.
#
# Usage:
# python auto_committer.py /path/to/your/repo feature.txt
#
# For testing (prevents actual commit and push):
# python auto_committer.py /path/to/your/repo feature.txt --dry-run
#
# How it works:
# 1. A 'stager' agent monitors the repo for file changes.
# 2. It uses tools to get the diff and analyze it with an LLM (Gemini)
#    against the feature description.
# 3. If the changes are relevant and complete, it stages them.
# 4. A 'committer' agent takes the staged diff, generates a commit
#    message using the LLM, and (if not in --dry-run)
#    commits and pushes the changes.
# 5. The script polls the repository at a defined interval.

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, TypedDict, Any

import git
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# --- 1. Load Environment Variables ---

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.", file=sys.stderr)
    sys.exit(1)

# --- 2. Define Tools for the Stager Agent ---
# These tools allow the agent to inspect the local repository.


@tool
def get_working_directory_diff(repo_path: str) -> str:
    """
    Gets all uncommitted, unstaged changes (the 'git diff')
    in the specified repository.
    """
    try:
        repo = git.Repo(repo_path)
        return repo.git.diff()
    except Exception as e:
        return f"Error getting diff: {e}"


@tool
def list_repository_files(repo_path: str) -> str:
    """
    Lists all files in the repository, respecting .gitignore.
    Returns a JSON string list.
    """
    try:
        repo = git.Repo(repo_path)
        files = repo.git.ls_files().splitlines()
        return json.dumps(files)
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def get_file_contents(repo_path: str, file_path: str) -> str:
    """
    Gets the current contents of a specific file from the working directory.
    """
    try:
        full_path = (Path(repo_path) / file_path).resolve()
        # Security check to prevent reading files outside the repo
        if not full_path.is_relative_to(Path(repo_path).resolve()):
            return "Error: File path is outside the repository."
        return full_path.read_text()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"


@tool
def stage_files(repo_path: str, files: List[str]) -> str:
    """
    Stages the specified list of files for the next commit.
    The 'files' argument must be a list of file paths.
    """
    try:
        repo = git.Repo(repo_path)
        # Verify files exist and are in the repo
        valid_files = []
        repo_root = Path(repo_path).resolve()
        for f in files:
            full_path = (repo_root / f).resolve()
            if full_path.is_relative_to(repo_root) and full_path.exists():
                valid_files.append(f)
            else:
                print(f"Warning: Skipping non-existent or outside file: {f}")
        
        if not valid_files:
            return "Error: No valid files provided to stage."

        repo.index.add(valid_files)
        return f"Successfully staged: {valid_files}"
    except Exception as e:
        return f"Error staging files: {e}"


# --- 3. Define Agent State ---
# This TypedDict holds the state that is passed between nodes in our graph.


class AgentState(TypedDict):
    repo_path: str
    feature_description: str
    dry_run: bool
    messages: list  # Conversation history for the stager agent
    staged_diff: str  # The diff of what's been staged
    commit_message: str
    push_status: str


# --- 4. Configure LLMs ---

# General-purpose LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Using 1.5-flash for reliability
    temperature=0,
)

# LLM for the stager agent, bound with our repository tools
stager_tools = [
    get_working_directory_diff,
    list_repository_files,
    get_file_contents,
    stage_files,
]
stager_agent_llm = llm.bind_tools(stager_tools)


# --- 5. Define Graph Nodes ---

def stager_agent_node(state: AgentState) -> dict:
    """
    This node represents the 'Stager Agent'. It uses the LLM with tools
    to analyze the repo and decide whether to stage files.
    """
    print("--- 🕵️ Stager Agent Running ---")
    repo_path = state["repo_path"]
    feature = state["feature_description"]

    system_prompt = f"""
    You are a 'Stager' agent. Your job is to monitor a code repository at
    '{repo_path}' for changes related to a specific feature.
    Your goal is to decide when the changes are complete and relevant,
    and then stage them for commit.

    Feature Description & Acceptance Criteria:
    ---
    {feature}
    ---

    You have the following tools:
    - `get_working_directory_diff(repo_path)`: Shows all uncommitted, unstaged changes.
    - `list_repository_files(repo_path)`: List all files in the repo.
    - `get_file_contents(repo_path, file_path)`: Get contents of a specific file.
    - `stage_files(repo_path, files)`: Stages the specified files.

    Your process:
    1.  Call `get_working_directory_diff()` with `repo_path='{repo_path}'`
        to see if there are any changes.
    2.  If the diff is empty, stop and report "No changes detected."
    3.  If there is a diff, analyze it against the feature description.
    4.  Decide if the changes are (a) relevant and (b) complete enough for a commit.
        You may use `get_file_contents` to inspect files further if needed.
    5.  If NOT ready, stop and report your reasoning
        (e.g., "Changes are partial...").
    6.  If READY, analyze the diff and identify ALL modified files that
        should be part of this commit.
    7.  Call `stage_files(repo_path='{repo_path}', files=[...])` with the
        list of files to stage. This is your final action.
        DO NOT call this tool with an empty list.
    """

    # Initialize messages for the agent
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Start monitoring and analysis."),
    ]

    # Run the agent loop (max 5 steps)
    for i in range(5):
        print(f"Stager Agent: Invoking LLM (Step {i+1}/5)...")
        try:
            response: AIMessage = stager_agent_llm.invoke(messages)
            messages.append(response)

            if response.tool_calls:
                print(f"Tool calls: {response.tool_calls}")
                for tc in response.tool_calls:
                    print(f"Stager Agent: Executing tool '{tc['name']}' with args: {tc['args']}")
                    # Execute the tool
                    tool_func = globals()[tc["name"]]
                    result = tool_func.invoke(tc["args"])
                    messages.append(
                        ToolMessage(content=str(result), tool_call_id=tc["id"])
                    )
                    # If we staged files, our job is done.
                    if tc["name"] == "stage_files":
                        print(f"Stager agent action: {result}")
                        return {"messages": messages}
            else:
                # Agent responded without a tool call
                print(f"Stager agent observation: {response.content}")
                
                # --- FIX 1 START ---
                # Cast content to string before calling .lower()
                content_str = str(response.content)

                # If agent says no changes or not ready, stop.
                if "no changes" in content_str.lower() or \
                   "not ready" in content_str.lower() or \
                   "partial" in content_str.lower():
                    print("Stager Agent: Observation indicates no action required. Stopping loop.")
                    return {"messages": messages}
                # --- FIX 1 END ---
        
        except Exception as e:
            print(f"Error in stager agent loop: {e}")
            messages.append(HumanMessage(content=f"An error occurred: {e}. Please reassess."))


    print("Stager agent finished cycle.")
    return {"messages": messages}


def committer_agent_node(state: AgentState) -> dict:
    """
    This node represents the 'Committer Agent'. It generates a commit
    message based on the staged diff.
    """
    print("--- ✍️ Committer Agent Running ---")
    staged_diff = state["staged_diff"]
    feature = state["feature_description"]
    
    # --- Check if diff is empty (which it shouldn't be if graph logic is correct) ---
    if not staged_diff:
        print("Committer Agent: ERROR - Received an empty diff. Aborting.")
        return {"commit_message": "error: no diff provided"}

    print(f"Committer Agent: Received staged diff of {len(staged_diff)} characters.")

    prompt = f"""
    Given the following staged changes (git diff):
    ---
    {staged_diff}
    ---

    And the feature context:
    ---
    {feature}
    ---

    Generate a concise, conventional commit message (e.g., 'feat: add new feature').
    Respond with ONLY the commit message string, no other text,
    markdown, or JSON.
    """
    
    try:
        response = llm.invoke(prompt)
        message = response.content.strip().strip("`")
        print(f"Generated commit message: {message}")
        return {"commit_message": message}
    except Exception as e:
        print(f"Error generating commit message: {e}")
        return {"commit_message": "fix: auto-commit generation failed"}


def commit_and_push_node(state: AgentState) -> dict:
    """
    This node performs the final git commit and push operations,
    unless --dry-run is specified.
    """
    print("--- 🚀 Commit and Push Node Running ---")
    if state["dry_run"]:
        print("\n--- DRY RUN ---")
        print("Changes are staged.")
        print(f"Would commit with message: '{state['commit_message']}'")
        print("Skipping actual commit and push.")
        return {"push_status": "dry_run_skipped"}

    try:
        repo = git.Repo(state["repo_path"])
        repo.index.commit(state["commit_message"])
        print("Changes committed successfully.")

        # Push to origin (assumes 'origin' remote and current branch)
        origin = repo.remote(name="origin")
        origin.push()
        print("Changes pushed to origin successfully.")
        return {"push_status": "pushed_successfully"}
    except Exception as e:
        print(f"Error during commit or push: {e}", file=sys.stderr)
        return {"push_status": f"Error: {e}"}


# --- FIX 2 START ---
# We replace the old conditional edge with a new node and a new conditional edge.

# --- 6. New Node to Check Staging ---

def check_staging_node(state: AgentState) -> dict:
    """
    Checks if files are staged and updates the state
    with the staged diff. This is a dedicated node to ensure
    state is updated correctly.
    """
    print("--- 🤔 Checking if files are staged ---")
    try:
        repo = git.Repo(state["repo_path"])
        staged_diff = repo.git.diff(staged=True)

        if not staged_diff:
            print("No files were staged. Ending cycle.")
            # Return the empty diff to update the state
            return {"staged_diff": ""}
        else:
            print(f"Staged changes detected ({len(staged_diff)} chars). Proceeding to commit.")
            # Return the found diff to update the state
            return {"staged_diff": staged_diff}
    except Exception as e:
        print(f"Error checking staged diff: {e}", file=sys.stderr)
        return {"staged_diff": ""} # Return empty diff on error


# --- 7. Define Graph Conditional Logic ---

def did_stage(state: AgentState) -> str:
    """
    This is a conditional edge. It checks if the
    check_staging_node found a diff in the state.
    """
    if state.get("staged_diff"):
        return "generate_commit_message"
    else:
        return END

# --- FIX 2 END ---


# --- 8. Build the Graph ---

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("stager_agent", stager_agent_node)
graph.add_node("check_staging", check_staging_node) # <-- FIX 2: Added new node
graph.add_node("committer_agent", committer_agent_node)
graph.add_node("commit_and_push", commit_and_push_node)

# Set the entry point
graph.set_entry_point("stager_agent")

# Add edges
graph.add_edge("stager_agent", "check_staging") # <-- FIX 2: Stager goes to check_staging

# <-- FIX 2: New conditional edge from check_staging
graph.add_conditional_edges(
    "check_staging",
    did_stage,
    {
        "generate_commit_message": "committer_agent",
        END: END,
    },
)
graph.add_edge("committer_agent", "commit_and_push")
graph.add_edge("commit_and_push", END)

# Compile the graph
app = graph.compile()


# --- 9. Main Execution Loop ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-Committer Agent System"
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Filesystem path to the local Git repository."
    )
    parser.add_argument(
        "feature_file",
        type=str,
        help="Path to a text file containing the feature description and AC."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps except the final git commit and push."
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Time in seconds to wait between checks (default: 60)."
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.repo_path).is_dir():
        print(f"Error: Repo path not found: {args.repo_path}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.feature_file).is_file():
        print(f"Error: Feature file not found: {args.feature_file}", file=sys.stderr)
        sys.exit(1)

    try:
        # Check if it's a valid git repo
        repo = git.Repo(args.repo_path)
    except git.exc.InvalidGitRepositoryError:
        print(f"Error: Not a valid git repository: {args.repo_path}", file=sys.stderr)
        sys.exit(1)

    # Read feature description
    feature_description = Path(args.feature_file).read_text()

    if args.dry_run:
        print("--- RUNNING IN DRY-RUN MODE ---")

    print("\n" + "="*50)
    print("Agent Graph Structure (Mermaid):")
    try:
        print(app.get_graph().draw_mermaid())
    except Exception as e:
        print(f"Could not draw graph: {e}")
    print("="*50 + "\n")

    # Define the initial state
    initial_state: AgentState = {
        "repo_path": args.repo_path,
        "feature_description": feature_description,
        "dry_run": args.dry_run,
        "messages": [],
        "staged_diff": "",
        "commit_message": "",
        "push_status": "",
    }

    # Start the polling loop
    try:
        while True:
            print("\n" + "="*50)
            print(f"[{time.ctime()}] Running check on {args.repo_path}...")
            
            # Create a fresh copy of the state for this run,
            # except for persistent fields if we had any.
            run_state = initial_state.copy()
            run_state["messages"] = [] # Reset messages for each run

            try:
                # Invoke the agent graph
                final_state = app.invoke(run_state)
                
                if final_state.get("push_status"):
                    print(f"Cycle complete. Status: {final_state['push_status']}")
                else:
                    print("Cycle complete. No action taken.")

            except Exception as e:
                print(f"An error occurred during graph execution: {e}", file=sys.stderr)

            print(f"Waiting for {args.poll_interval} seconds...")
            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print("\nPolling stopped by user. Exiting.")
        sys.exit(0)