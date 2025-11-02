import operator
import os
import json
import time
import requests
from typing import TypedDict, Annotated, Literal, Dict, List, Tuple
from dotenv import load_dotenv

# --- Load Environment Variables ---
# NOTE: Ensure you have SLACK_BOT_TOKEN, OPENROUTER_API_KEY, 
# GITHUB_TOKEN, REPO_OWNER, and REPO_NAME set in your .env file
load_dotenv()

# --- User's Existing Slack Agent Setup ---

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage, AIMessage
from langchain_core.tools.structured import StructuredTool
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# model config
model = ChatOpenAI(
  api_key=os.getenv("OPENROUTER_API_KEY"),
  base_url="https://openrouter.ai/api/v1",
  model="anthropic/claude-3.5-sonnet"
)

# define tools
client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))

def send_slack_message(recipient: str, message: str) -> str:
    """Sends a message to a Slack user or channel."""
    try:
        response = client.chat_postMessage(
            channel=recipient,
            text=message
        )
        return f"Message sent successfully to {recipient}."
    except SlackApiError as e:
        return f"Error sending message: {e.response['error']}"
    
send_message_tool = StructuredTool.from_function(
    func=send_slack_message,
    name="send_slack_message",
    description="Send a Slack message to a specific user or channel. The recipient must be a valid user ID, username (e.g., @john_doe), or channel name (e.g., #general)."
)

tools = [send_message_tool]
llm_with_tools = model.bind_tools(tools)

# Define the State Schema
class SlackWriterState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# Define Agent Nodes
def llm_call(state: SlackWriterState) -> SlackWriterState:
    """The LLM node: calls the model to decide on an action."""
    system_prompt = (
        "You are an automated Slack assistant tasked with sending messages. "
        "If the user explicitly asks you to 'send' a message, you MUST call "
        "the 'send_slack_message' tool. Always be concise."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]} 

def tool_node(state: SlackWriterState) -> SlackWriterState:
    """The Tool node: executes the tool call requested by the LLM."""
    last_message = state["messages"][-1]
    tool_messages = []
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]

            if tool_name == send_message_tool.name:
                observation = send_slack_message(**tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call_id
                    )
                )
    
    return {"messages": tool_messages}

# Define Conditional Edge Function
def route_decision(state: SlackWriterState) -> Literal["tool_node", END]:
    """Decides the next step: either call the tool or end the graph."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    return END

# Build and Compile Graph
graph_builder = StateGraph(SlackWriterState)
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call",
    route_decision,
    {"tool_node": "tool_node", END: END}
)
graph_builder.add_edge("tool_node", "llm_call")
agent = graph_builder.compile()

def run_agent(user_input: str) -> str:
    """Invokes the compiled Slack agent with a message."""
    initial_message = [HumanMessage(content=user_input)]
    # Suppressing internal print statements to keep the output clean
    result = agent.invoke({"messages": initial_message})
    final_message = result["messages"][-1].content
    return final_message

# --- New GitHub Agent Logic ---

def parse_plan(plan_content: str) -> Dict[str, List[str]]:
    """
    Parses the plan.json content to extract target commit titles for each branch.
    """
    plan_data = json.loads(plan_content)
    target_commits_by_branch = {}

    for branch_planset in plan_data.get("person_to_branchplanset", {}).values():
        for branch_plan in branch_planset.get("branches", []):
            branch_name = branch_plan["name"]
            target_commit_titles = [commit["title"] for commit in branch_plan["commits"]]
            target_commits_by_branch[branch_name] = target_commit_titles
            
    return target_commits_by_branch

def check_github_commits(
    repo_owner: str, 
    repo_name: str, 
    target_commits_by_branch: Dict[str, List[str]],
    last_processed_commit_sha: Dict[str, str],
    github_token: str
) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    """
    Checks GitHub for new commits on target branches and identifies completed tasks.
    
    Returns: (updated_last_processed_sha, completed_commits)
    """
    completed_commits = []
    updated_last_processed_sha = last_processed_commit_sha.copy()
    headers = {"Authorization": f"token {github_token}"}
    
    for branch_name, target_titles_list in target_commits_by_branch.items():
        
        # 1. Fetch latest commit for the branch
        branch_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches/{branch_name}"
        try:
            response = requests.get(branch_url, headers=headers)
            response.raise_for_status()
            latest_sha = response.json()["commit"]["sha"]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching branch {branch_name}: {e}")
            continue

        last_sha = updated_last_processed_sha.get(branch_name)

        if not last_sha:
            # First run: Initialize SHA and skip commit processing
            updated_last_processed_sha[branch_name] = latest_sha
            print(f"First run for branch {branch_name}. Initialized SHA: {latest_sha}. Skipping commit check.")
            continue
            
        if latest_sha == last_sha:
            continue
            
        # 2. Fetch commits between last_sha and latest_sha
        # Fetching a list of commits for the branch
        commits_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits?sha={branch_name}&per_page=100"
        try:
            commits_response = requests.get(commits_url, headers=headers)
            commits_response.raise_for_status()
            commits_data = commits_response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching commits for {branch_name}: {e}")
            continue

        # 3. Process new commits
        commits_to_process = []
        for commit in commits_data:
            if commit["sha"] == last_sha:
                break
            commits_to_process.append(commit)
            
        commits_to_process.reverse() # Process oldest to newest
        
        target_titles_set = set(target_titles_list)
        
        for commit in commits_to_process:
            # Use the first line of the commit message as the title
            commit_title = commit["commit"]["message"].split('\n')[0].strip()
            
            if commit_title in target_titles_set:
                completed_commits.append({
                    "branch": branch_name,
                    "title": commit_title,
                    "sha": commit["sha"][:7],
                    "url": commit["html_url"],
                    "author": commit["commit"]["author"]["name"]
                })
                # Remove the title to prevent re-triggering for the same commit
                target_titles_set.discard(commit_title)
                
        # 4. Update the processed SHA to the latest SHA
        if commits_to_process:
            updated_last_processed_sha[branch_name] = latest_sha

    return updated_last_processed_sha, completed_commits


def run_github_slack_loop(
    plan_content: str, 
    repo_owner: str, 
    repo_name: str, 
    github_token: str,
    interval_seconds: int = 60
):
    """Main loop to check GitHub and trigger the Slack Agent."""
    
    target_commits_by_branch = parse_plan(plan_content)
    last_processed_commit_sha = {} # SHA tracker for each branch

    print("--- GitHub/Slack Agent Started ---")
    print(f"Agent will check for commit titles that complete tasks from: {list(target_commits_by_branch.keys())}")
    
    if not github_token:
        print("\nERROR: GITHUB_TOKEN environment variable not set. Exiting.")
        return

    while True:
        try:
            # 1. Check for new commits
            updated_sha, completed_commits = check_github_commits(
                repo_owner, 
                repo_name, 
                target_commits_by_branch, 
                last_processed_commit_sha,
                github_token
            )
            
            last_processed_commit_sha = updated_sha
            
            # 2. Process completed tasks
            if completed_commits:
                print(f"\n✅ Found {len(completed_commits)} new completed tasks. Triggering Slack notifications.")
                
                for commit_data in completed_commits:
                    branch = commit_data['branch']
                    title = commit_data['title']
                    sha = commit_data['sha']
                    url = commit_data['url']
                    author = commit_data['author']
                    
                    message = (
                        f"A planned task is complete! 🎉\n\n"
                        f"➡️ *Task:* `{title}`\n"
                        f"👤 *Author:* {author}\n"
                        f"🌿 *Branch:* `{branch}`\n"
                        f"🔗 *Commit:* <{url}|{sha}>\n\n"
                        f"Great work!"
                    )
                    
                    # 3. Trigger Slack Agent
                    slack_input = f"send '{message}' to #code"
                    print(f"   - Triggering Slack Agent for commit {sha}...")
                    slack_response = run_agent(slack_input)
                    print(f"   - Slack Agent Response: {slack_response}")
                    
            else:
                print(".", end="", flush=True) # Print a dot for status check
                
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            
        # 4. Wait for the next interval
        time.sleep(interval_seconds)


# --- Execution Block ---

# 1. Store the content of your plan.json
PLAN_CONTENT = """
{
  "person_to_branchplanset": {
    "@maxbachmann": {
      "branches": [
        {
          "name": "feature/weighted-edits",
          "rationale": "To enhance similarity scoring by allowing certain character substitutions to count differently in edit distance calculations.",
          "tasks_included": [
            "task_1"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Implement distance_with_weights method"
            ],
            "src/Levenshtein/__init__.py": [
              "Expose distance_with_weights in the API"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for distance_with_weights"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for distance_with_weights"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Implement distance_with_weights method",
              "body": "Added a new method to calculate edit distance with weighted character substitutions.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added distance_with_weights method"
              ]
            },
            {
              "title": "Expose distance_with_weights in the API",
              "body": "Updated the __init__.py to include the new method in the public API.",
              "files": [
                "src/Levenshtein/__init__.py"
              ],
              "per_file_changes": [
                "Exposed distance_with_weights"
              ]
            },
            {
              "title": "Update documentation for distance_with_weights",
              "body": "Documented the new distance_with_weights method in the Levenshtein documentation.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Updated documentation for distance_with_weights"
              ]
            },
            {
              "title": "Add tests for distance_with_weights",
              "body": "Added comprehensive test cases for the distance_with_weights method.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added tests for distance_with_weights"
              ]
            }
          ]
        },
        {
          "name": "feature/pure-python-fallback",
          "rationale": "To provide a pure-Python fallback implementation for platforms where C extensions cannot be compiled, ensuring broader compatibility.",
          "tasks_included": [
            "task_2"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_py.dart": [
              "Create pure-Python fallback module"
            ],
            "src/Levenshtein/__init__.py": [
              "Add logic to select fallback module"
            ],
            "docs/levenshtein.rst": [
              "Document fallback module and performance trade-off"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Create fallback module",
              "body": "Added a pure-Python fallback module for platforms without C extension support.",
              "files": [
                "src/Levenshtein/levenshtein_py.dart"
              ],
              "per_file_changes": [
                "Created levenshtein_py.dart fallback module"
              ]
            },
            {
              "title": "Add selection logic for fallback module",
              "body": "Updated __init__.py to include logic for selecting the fallback module.",
              "files": [
                "src/Levenshtein/__init__.py"
              ],
              "per_file_changes": [
                "Added logic to select fallback module"
              ]
            },
            {
              "title": "Document fallback module",
              "body": "Documented the fallback module and its performance trade-off in the Levenshtein documentation.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for fallback module"
              ]
            }
          ]
        }
      ]
    }
  }
}
"""
PLAN_CONTENT = """
{
  "person_to_branchplanset": {
    "@maxbachmann": {
      "branches": [
        {
          "name": "feature/weighted-edits",
          "rationale": "To enhance similarity scoring by allowing certain character substitutions to count differently in edit distance calculations.",
          "tasks_included": [
            "task_1"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Add new API method distance_with_weights"
            ],
            "src/Levenshtein/__init__.py": [
              "Update API to include distance_with_weights"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for distance_with_weights"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for distance_with_weights"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Implement distance_with_weights method",
              "body": "Added a new method to calculate edit distance with weighted character substitutions.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added distance_with_weights function"
              ]
            },
            {
              "title": "Update API for weighted edits",
              "body": "Updated the __init__.py to expose the new distance_with_weights method.",
              "files": [
                "src/Levenshtein/__init__.py"
              ],
              "per_file_changes": [
                "Exposed distance_with_weights in the API"
              ]
            },
            {
              "title": "Update documentation for distance_with_weights",
              "body": "Updated the documentation to include the new method and its usage.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for distance_with_weights"
              ]
            },
            {
              "title": "Add tests for distance_with_weights",
              "body": "Created unit tests to validate the functionality of distance_with_weights.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added test cases for distance_with_weights"
              ]
            }
          ]
        },
        {
          "name": "feature/max-distance-cutoff",
          "rationale": "To improve performance by allowing the distance method to return early for inputs exceeding a specified threshold.",
          "tasks_included": [
            "task_2"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Add new methods distance(s1, s2, max_dist=N) and similarity(s1, s2, max_dist=N)"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for new max distance methods"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for max distance methods"
            ]
          },
          "risk_notes": [
            "Ensure backward compatibility with existing methods."
          ],
          "commits": [
            {
              "title": "Implement max distance methods",
              "body": "Added distance and similarity methods with max distance parameter.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added distance(s1, s2, max_dist=N) and similarity(s1, s2, max_dist=N)"
              ]
            },
            {
              "title": "Update documentation for max distance methods",
              "body": "Updated the documentation to include the new max distance methods.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for distance and similarity with max distance"
              ]
            },
            {
              "title": "Add tests for max distance methods",
              "body": "Created unit tests to validate the functionality of the new max distance methods.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added test cases for distance and similarity with max distance"
              ]
            }
          ]
        }
      ]
    },
    "@guyrosin": {
      "branches": [
        {
          "name": "feature/multi-threaded-batch",
          "rationale": "To allow faster processing of large lists of strings for similarity ranking through parallel computation.",
          "tasks_included": [
            "task_3"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Add new batch method batch_distance(list1, list2, threads=T)"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for batch_distance method"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for batch_distance method"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Implement batch_distance method",
              "body": "Added a new method to compute distances in batches using multiple threads.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added batch_distance function"
              ]
            },
            {
              "title": "Update documentation for batch_distance",
              "body": "Updated the documentation to include the new batch_distance method.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for batch_distance"
              ]
            },
            {
              "title": "Add tests for batch_distance method",
              "body": "Created unit tests to validate the functionality of batch_distance.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added test cases for batch_distance"
              ]
            }
          ]
        },
        {
          "name": "feature/edit-operations-path",
          "rationale": "To provide detailed insights into the edit operations performed during distance calculations.",
          "tasks_included": [
            "task_4"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Add new method distance_with_ops(s1, s2)"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for distance_with_ops method"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for distance_with_ops method"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Implement distance_with_ops method",
              "body": "Added a new method to output the edit operations path.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added distance_with_ops function"
              ]
            },
            {
              "title": "Update documentation for distance_with_ops",
              "body": "Updated the documentation to include the new distance_with_ops method.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for distance_with_ops"
              ]
            },
            {
              "title": "Add tests for distance_with_ops method",
              "body": "Created unit tests to validate the functionality of distance_with_ops.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added test cases for distance_with_ops"
              ]
            }
          ]
        }
      ]
    },
    "@antoinetavant": {
      "branches": [
        {
          "name": "feature/unicode-grapheme-clusters",
          "rationale": "To ensure multi-codepoint characters are treated as single units in distance calculations, improving correctness.",
          "tasks_included": [
            "task_5"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_cpp.pyx": [
              "Add new optional parameter use_graphemes=True"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for grapheme support"
            ],
            "tests/test_levenshtein_distance.py": [
              "Add tests for grapheme support"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Implement grapheme support",
              "body": "Added an optional parameter to support grapheme clusters.",
              "files": [
                "src/Levenshtein/levenshtein_cpp.pyx"
              ],
              "per_file_changes": [
                "Added use_graphemes parameter"
              ]
            },
            {
              "title": "Update documentation for grapheme support",
              "body": "Updated the documentation to include the new grapheme support.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for grapheme support"
              ]
            },
            {
              "title": "Add tests for grapheme support",
              "body": "Created unit tests to validate the functionality of grapheme support.",
              "files": [
                "tests/test_levenshtein_distance.py"
              ],
              "per_file_changes": [
                "Added test cases for grapheme support"
              ]
            }
          ]
        }
      ]
    },
    "@LecrisUT": {
      "branches": [
        {
          "name": "feature/pure-python-fallback",
          "rationale": "To maintain functionality on platforms where the C extension cannot compile, enhancing portability.",
          "tasks_included": [
            "task_6"
          ],
          "parts_covered": {
            "src/Levenshtein/levenshtein_py.dart": [
              "Add fallback module"
            ],
            "src/Levenshtein/__init__.py": [
              "Update API to select fallback module"
            ],
            "docs/levenshtein.rst": [
              "Update documentation for fallback module"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Add pure-Python fallback module",
              "body": "Created a fallback module to ensure functionality on non-C compatible platforms.",
              "files": [
                "src/Levenshtein/levenshtein_py.dart"
              ],
              "per_file_changes": [
                "Added levenshtein_py.dart as a fallback"
              ]
            },
            {
              "title": "Update API for fallback selection",
              "body": "Updated the __init__.py to include logic for selecting the fallback module.",
              "files": [
                "src/Levenshtein/__init__.py"
              ],
              "per_file_changes": [
                "Added fallback selection logic"
              ]
            },
            {
              "title": "Update documentation for fallback module",
              "body": "Updated the documentation to include the new fallback module.",
              "files": [
                "docs/levenshtein.rst"
              ],
              "per_file_changes": [
                "Added documentation for fallback module"
              ]
            }
          ]
        },
        {
          "name": "docs/readme-review-note",
          "rationale": "To add a simple line in readme file that says 'reviewed by Bob'.",
          "tasks_included": [
            "task_7"
          ],
          "parts_covered": {
            "README.md": [
              "Add 'reviewed by Bob' line"
            ]
          },
          "risk_notes": [],
          "commits": [
            {
              "title": "Docs: Add review acknowledgement",
              "body": "Added 'reviewed by Bob' line to README.",
              "files": [
                "README.md"
              ],
              "per_file_changes": [
                "Added 'reviewed by Bob' line"
              ]
            }
          ]
        }
      ]
    }
  }
}"""

# 2. Get GitHub configuration from environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER = "nawazia"
REPO_NAME = "Levenshtein"

if __name__ == "__main__":
    if not GITHUB_TOKEN or not REPO_OWNER or not REPO_NAME:
        print("ERROR: Please set GITHUB_TOKEN, REPO_OWNER, and REPO_NAME in your .env file.")
    else:
        # Run the agent check loop every 60 seconds
        run_github_slack_loop(PLAN_CONTENT, REPO_OWNER, REPO_NAME, GITHUB_TOKEN, interval_seconds=60)
