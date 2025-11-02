import json
import os
import time
from typing import Annotated, Any, List, TypedDict
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agents.slack_writer import run_agent

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openai/gpt-3.5-turbo"
REPO = os.getenv("GITHUB_REPO_URL")

# --- 1. LangGraph State Definition ---


class AgentState(TypedDict):
    """
    Represents the state of the multi-agent system.
    """

    repo_url: str  # The input GitHub repository URL
    owner: str  # Extracted repository owner
    repo_name: str  # Extracted repository name

    # List of unique contributor usernames
    developers: List[str]

    # Map: dev_name -> raw contribution data (commits/files)
    contributions: dict[str, Any]

    # Map: dev_name -> LLM drafted profile text
    profiles_in_progress: dict[str, str]

    # Final output profiles (Map: dev_name -> final profile)
    final_profiles: dict[str, str]

    # State management for the loop
    developer_to_process: str  # The current developer being processed
    remaining_developers: List[str]  # Developers left to process


# --- 2. Tool/LLM Abstractions ---


def _call_openrouter(system_prompt: str, user_query: str) -> str:
    """
    Calls the OpenRouter API with the given prompt and query.
    Implements simple retry logic for robustness.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable not set.")
        return "LLM_CALL_FAILED: API key missing."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    }

    # Simple exponential backoff retry loop
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                print(
                    f"OpenRouter call failed (Attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s. Error: {e}"
                )
                time.sleep(wait_time)
            else:
                print(f"OpenRouter call failed after {max_retries} attempts: {e}")
                return "LLM_CALL_FAILED: Exhausted all retries."

    return "LLM_CALL_FAILED: Unknown error."


def _fetch_github_data(owner: str, repo_name: str) -> dict[str, Any]:
    """
    Fetches the last 100 commits from the GitHub repository API.
    A GitHub token is used for better rate limits.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    api_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?per_page=100"

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        commits = response.json()

        if not isinstance(commits, list) or response.status_code == 404:
            return {"error": "Repository not found or API limits exceeded."}

        # Structure contribution data per developer
        contributions = {}
        developers = set()

        for commit in commits:
            author_login = commit.get("author", {}).get("login")
            if not author_login:
                continue  # Skip commits without a valid GitHub account

            developers.add(author_login)

            # Fetch detailed commit view (files changed)
            commit_detail_url = commit["url"]
            detail_response = requests.get(
                commit_detail_url, headers=headers, timeout=10
            )
            detail_response.raise_for_status()
            detail_data = detail_response.json()

            # Initialize developer entry if needed
            if author_login not in contributions:
                contributions[author_login] = {"total_commits": 0, "files_touched": []}

            contributions[author_login]["total_commits"] += 1

            # Extract file paths from the detailed commit
            for file_change in detail_data.get("files", []):
                contributions[author_login]["files_touched"].append(
                    {
                        "filename": file_change.get("filename"),
                        "additions": file_change.get("additions"),
                        "deletions": file_change.get("deletions"),
                    }
                )

        return {"developers": list(developers), "contributions": contributions}

    except requests.exceptions.RequestException as e:
        return {"error": f"GitHub API request failed: {e}"}


# --- 3. LangGraph Nodes (Agents) ---


def parse_input(state: AgentState) -> AgentState:
    """Node 1: Parses the GitHub URL to extract owner and repo name."""
    repo_url = state["repo_url"]
    print(f"\n--- Node 1: Parsing URL: {repo_url} ---")

    try:
        parsed_url = urlparse(repo_url)
        path_parts = [p for p in parsed_url.path.split("/") if p]

        if parsed_url.netloc != "github.com" or len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL format.")

        owner = path_parts[0]
        repo_name = path_parts[1].replace(".git", "")

        return {
            **state,
            "owner": owner,
            "repo_name": repo_name,
            "profiles_in_progress": {},
            "final_profiles": {},
        }
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return {**state, "error": f"URL parsing failed: {e}"}


def fetch_contributor_data(state: AgentState) -> AgentState:
    """Node 2: Fetches raw contribution data from GitHub API."""
    print("\n--- Node 2: Fetching Contributor Data ---")
    owner = state["owner"]
    repo_name = state["repo_name"]

    data = _fetch_github_data(owner, repo_name)

    if "error" in data:
        print(f"Error fetching data: {data['error']}")
        return {**state, "error": data["error"]}

    developers = data["developers"]
    contributions = data["contributions"]

    print(f"Found {len(developers)} contributors: {', '.join(developers)}")

    # Initialize the loop state
    developer_to_process = developers[0] if developers else ""
    remaining_developers = developers[1:] if developers else []

    return {
        **state,
        "developers": developers,
        "contributions": contributions,
        "developer_to_process": developer_to_process,
        "remaining_developers": remaining_developers,
    }


def analyze_single_developer(state: AgentState) -> AgentState:
    """
    Node 3 (Looped Agent): Uses the LLM to analyze a single developer's contributions.
    """
    developer_id = state["developer_to_process"]
    contributions = state["contributions"].get(developer_id, {})

    print(f"\n--- Node 3: Analyzing Developer: {developer_id} ---")

    if not contributions:
        profile_draft = "Could not find detailed contribution data for this user."
    else:
        # Prepare the data for the LLM
        contribution_summary = json.dumps(contributions, indent=2)

        system_prompt = (
            "You are a Senior Development Analyst. Your task is to generate a comprehensive 'Dev Profile' "
            "for a collaborator based *only* on the provided raw contribution data. "
            "Analyze the file paths, additions, and deletions to infer their expertise. "
            "Output the profile in a structured, concise paragraph."
        )

        user_query = (
            f"Analyze the contributions for developer '{developer_id}' on this GitHub repository. "
            "Based on the data, identify their core expertise, preferred language/framework, and likely role (e.g., Front-end, Back-end, DevOps, Documentation, Full-Stack). "
            "Here is the raw data:\n\n"
            f"{contribution_summary}"
        )

        profile_draft = _call_openrouter(system_prompt, user_query)

    # Update state with the new profile draft
    profiles_in_progress = state["profiles_in_progress"]
    profiles_in_progress[developer_id] = profile_draft

    # Manage the loop: move to the next developer
    remaining = state["remaining_developers"]
    next_developer = remaining[0] if remaining else ""
    remaining_developers = remaining[1:] if remaining else []

    return {
        **state,
        "profiles_in_progress": profiles_in_progress,
        "developer_to_process": next_developer,
        "remaining_developers": remaining_developers,
    }


def aggregate_profiles(state: AgentState) -> AgentState:
    """Node 4: Collects all profiles and formats the final output."""
    print("\n--- Node 4: Aggregating Final Profiles ---")

    profiles = state["profiles_in_progress"]

    # Simple aggregation: just format the collected profiles
    final_output = {}
    report = "### Repository Contributor Profiles Report\n\n"
    report += f"Analyzed Repository: {state['repo_url']} ({state['owner']}/{state['repo_name']})\n\n"
    report += "---"

    for dev, profile in profiles.items():
        report += f"\n\n## Developer: @{dev}\n"
        report += profile
        final_output[dev] = profile

    print("Aggregation complete. Profiles generated.")

    return {**state, "final_profiles": final_output}


# --- 4. LangGraph Control Flow (Router) ---


def should_continue_loop(state: AgentState) -> str:
    """Router: Determines if there are more developers to process."""
    if state["developer_to_process"]:
        return "continue"
    else:
        return "end"


# --- 5. Graph Definition and Compilation ---

# Build the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("fetch_contributor_data", fetch_contributor_data)
workflow.add_node("analyze_single_developer", analyze_single_developer)
workflow.add_node("aggregate_profiles", aggregate_profiles)

# Define Edges (Flow)
workflow.set_entry_point("parse_input")

workflow.add_edge("parse_input", "fetch_contributor_data")

# Branching point after fetching data: check if any developers were found
workflow.add_conditional_edges(
    "fetch_contributor_data",
    should_continue_loop,
    {
        "continue": "analyze_single_developer",
        "end": "aggregate_profiles",  # Skip analysis if no developers found
    },
)

# Loop: from analysis, back to the router
workflow.add_conditional_edges(
    "analyze_single_developer",
    should_continue_loop,
    {
        "continue": "analyze_single_developer",  # Process next developer
        "end": "aggregate_profiles",  # All developers processed
    },
)

# Final step
workflow.add_edge("aggregate_profiles", END)

# Compile the graph
app = workflow.compile()


# --- 6. Execution ---


def run_analysis(repo_url: str):
    """Executes the LangGraph application."""
    print(f"Starting Dev Profile Analysis for: {repo_url}")

    initial_state = {
        "repo_url": repo_url,
        "owner": "",
        "repo_name": "",
        "developers": [],
        "contributions": {},
        "profiles_in_progress": {},
        "final_profiles": {},
        "developer_to_process": "",
        "remaining_developers": [],
    }

    # Running the graph
    final_state = app.invoke(initial_state)

    if final_state.get("error"):
        print("\n\n--- Analysis Failed ---")
        print(f"Error: {final_state['error']}")
        return

    print("\n\n--- Final Analysis Report (Generated in Markdown) ---")

    # We will format the final output using a markdown block for readability
    print("Generating report file...")
    report_content = final_state["final_profiles"]

    markdown_report = f"# Contributor Profile Analysis for {repo_url}"

    if not report_content:
        markdown_report += "\n\nNo contributors found or data fetching failed."
    else:
        for dev, profile in report_content.items():
            markdown_report += f"\n\n---\n\n## 👤 Developer: @{dev}\n\n"
            markdown_report += profile

    print(markdown_report)  # Print the raw markdown for console review

    return markdown_report


def main():
    # Run the analysis
    final_report = run_analysis(REPO)

    # Generate the output file for the report
    if final_report:
        # Use a temporary name based on the repo for the markdown file
        repo_name_safe = REPO.split("/")[-1].replace(".", "_")
        filepath = f"dev_profiles_{repo_name_safe}.md"

        run_agent(
            "Send on slack to #all-agentsverse-hackathon: DEVELOPER PROFILES GENERATED"
        )
        # run_agent(
        #     "Format and send the following to #all-agentsverse-hackathon:"
        #     + final_report
        # )

        print(final_report)
