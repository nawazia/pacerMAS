import os
import time
import git  # Import the gitpython library
import argparse # <-- 1. IMPORTED ARGPARSE
from dotenv import load_dotenv
from typing import TypedDict, Literal, Optional

from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ------------------------------------------------------------------
# 1. ⚠️ CONFIGURATION: SET YOUR KEYS AND PATHS HERE ⚠️
# ------------------------------------------------------------------

# 1. Set your Google API key (get one from https://aistudio.google.com/)
# Already loaded in .env file


# 2. Set the *full* path to the local Git repository you want to monitor
#    Example for Windows: "C:\\Users\\YourName\\Projects\\my-repo"
#    Example for macOS/Linux: "/Users/yourname/projects/my-repo"
REPO_PATH = "/Users/eansengchang/Documents/Stuff/projects/hackathons/pacerMAS"

# 3. Describe the feature you are currently working on
#    The AI will use this to decide if changes are relevant.
FEATURE_DESCRIPTION = (
    "Implement a new auto committer "
    "This includes automatically detecting if a new commit is necessary and then automatically committing with a relevant message"
)

# 4. Set how often to check for changes (in seconds)
CHECK_INTERVAL_SECONDS = 30

# ------------------------------------------------------------------
# 2. DEFINE THE STATE FOR OUR AGENT SYSTEM
# ------------------------------------------------------------------

class AgentState(TypedDict):
    """Defines the state that flows through the graph."""
    repo_path: str
    feature_description: str
    wait_interval_seconds: int
    changes: str  # The full 'git diff' output
    commit_decision: Literal["yes", "no", ""] # AI's decision
    commit_message: str # AI-generated commit message
    dry_run: bool # <-- 2. ADDED DRY_RUN TO STATE
    error: Optional[str] # To hold any errors

# ------------------------------------------------------------------
# 3. DEFINE HELPER FUNCTIONS (GIT OPERATIONS)
# ------------------------------------------------------------------

def get_git_diff(repo_path: str) -> tuple[str, Optional[str]]:
    """
    Checks for staged, unstaged, and untracked changes in the repo.
    Returns a combined diff string and an error string, if any.
    """
    try:
        repo = git.Repo(repo_path)
        
        # 1. Get staged changes (diff against HEAD)
        diff_staged = repo.git.diff("--staged")
        
        # 2. Get unstaged changes (diff against index)
        diff_unstaged = repo.git.diff()
        
        # 3. Get list of untracked files
        untracked_files = repo.untracked_files
        untracked_str = "\n".join([f"Untracked file: {f}" for f in untracked_files])
        
        # Combine all changes into one string
        full_diff = (
            f"--- Staged Changes ---\n{diff_staged}\n\n"
            f"--- Unstaged Changes ---\n{diff_unstaged}\n\n"
            f"--- Untracked Files ---\n{untracked_str}"
        )
        
        # Check if there are any changes at all
        if not diff_staged and not diff_unstaged and not untracked_files:
            return "", None  # No changes
            
        return full_diff, None
        
    except git.exc.InvalidGitRepositoryError:
        return "", f"Error: '{repo_path}' is not a valid Git repository."
    except Exception as e:
        return "", f"Error getting Git diff: {e}"

def perform_git_commit(repo_path: str, message: str) -> Optional[str]:
    """
    Stages all changes and performs a commit with the given message.
    Returns an error string, if any.
    """
    try:
        repo = git.Repo(repo_path)
        
        # Stage all changes (add untracked, modified, and deleted)
        repo.git.add(A=True)
        
        # Perform the commit
        repo.index.commit(message)
        
        print(f"\n✅ Successfully committed with message:\n{message}")
        return None
    except Exception as e:
        print(f"\n❌ Error during commit: {e}")
        return f"Error during commit: {e}"

# ------------------------------------------------------------------
# 4. SET UP THE LLM (GEMINI)
# ------------------------------------------------------------------

# We'll use Gemini Pro, which is great for these tasks
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

# ------------------------------------------------------------------
# 5. DEFINE THE GRAPH NODES (THE AGENTS' ACTIONS)
# ------------------------------------------------------------------

def check_changes_node(state: AgentState) -> dict:
    """Node 1: Checks for file changes in the repository."""
    print("... Monitoring for changes ...")
    
    diff, error = get_git_diff(state['repo_path'])
    
    if error:
        print(error)
        return {"error": error, "changes": ""}
        
    if not diff:
        # No changes found.
        return {"changes": "", "commit_decision": ""}
    
    # Changes were found!
    print("🔥 Changes detected!")
    return {"changes": diff, "error": None}


def decide_to_commit_node(state: AgentState) -> dict:
    """Node 2: (Agent 1) AI decides if the changes are ready for commit."""
    print("... AI deciding if changes are commit-worthy ...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a senior software engineer. Your job is to decide if the current "
         "code changes are substantial enough and aligned with the planned feature "
         "to warrant a commit. Do not commit if the work is clearly half-finished "
         "(e.g., functions defined but not used, 'TODO' comments for core logic)."),
        ("human",
         "Here is the planned feature:\n<feature>\n{feature_description}\n</feature>\n\n"
         "Here are the current code changes (diff):\n<changes>\n{changes}\n</changes>\n\n"
         "Based on these changes and the feature, is it a good time to commit?\n"
         "Answer with only 'yes' or 'no'.")
    ])
    
    decide_chain = prompt | llm | StrOutputParser()
    
    result = decide_chain.invoke({
        "feature_description": state['feature_description'],
        "changes": state['changes']
    })
    
    decision = "yes" if "yes" in result.lower() else "no"
    print(f"... AI decision: {decision.upper()} ...")
    return {"commit_decision": decision}


def generate_commit_message_node(state: AgentState) -> dict:
    """Node 3: (Agent 2) AI generates a commit message for the changes."""
    print("... AI generating commit message ...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert at writing high-quality, concise git commit messages. "
         "Generate a commit message in the conventional commit format "
         "(e.g., 'feat: add new login endpoint'). The message should be based "
         "on the provided code diff and feature description."),
        ("human",
         "Feature Description:\n{feature_description}\n\n"
         "Code Diff:\n{changes}\n\n"
         "Generate the conventional commit message (header and optional body).")
    ])
    
    message_chain = prompt | llm | StrOutputParser()
    
    message = message_chain.invoke({
        "feature_description": state['feature_description'],
        "changes": state['changes']
    })
    
    return {"commit_message": message.strip().strip('`')}


def commit_node(state: AgentState) -> dict:
    """Node 4: Performs the actual git commit operation (or simulates it)."""
    # <-- 3. MODIFIED COMMIT_NODE -->
    
    if state['dry_run']:
        print("... [DRY RUN] Simulating commit ...")
        print(f"--- [DRY RUN] Commit Message Would Be: ---")
        print(state['commit_message'])
        print("-------------------------------------------")
        error = None # Simulate successful commit
    else:
        print("... Attempting to commit changes ...")
        error = perform_git_commit(state['repo_path'], state['commit_message'])
    
    if error:
        return {"error": error}
        
    # Clear state after successful commit (or simulated commit)
    return {
        "error": None, 
        "changes": "", 
        "commit_message": "", 
        "commit_decision": ""
    }


def wait_node(state: AgentState) -> dict:
    """Node 5: Waits for a set interval before checking again."""
    wait_interval = state['wait_interval_seconds']
    print(f"... Waiting for {wait_interval} seconds ...")
    time.sleep(wait_interval)
    return {} # No state change, just pauses execution

# ------------------------------------------------------------------
# 6. DEFINE THE GRAPH EDGES (THE LOGIC FLOW)
# ------------------------------------------------------------------

def after_check_changes(state: AgentState) -> Literal["decide_to_commit", "wait"]:
    """Conditional Edge 1: After checking, either decide or wait."""
    if state.get("error"):
        print("Error detected, waiting before retry.")
        return "wait" # Wait and retry if there was a git error
    if state["changes"]:
        return "decide_to_commit" # Changes found, let AI decide
    else:
        return "wait" # No changes, go to wait node

def after_decide_commit(state: AgentState) -> Literal["generate_message", "wait"]:
    """Conditional Edge 2: After deciding, either commit or wait."""
    if state["commit_decision"] == "yes":
        return "generate_message" # AI said yes, generate message
    else:
        return "wait" # AI said no, wait for more changes

# ------------------------------------------------------------------
# 7. BUILD AND COMPILE THE LANGGRAPH WORKFLOW
# ------------------------------------------------------------------

workflow = StateGraph(AgentState)

# Add all the nodes
workflow.add_node("check_changes", check_changes_node)
workflow.add_node("decide_to_commit", decide_to_commit_node)
workflow.add_node("generate_message", generate_commit_message_node)
workflow.add_node("commit_changes", commit_node)
workflow.add_node("wait", wait_node)

# Set the entry point
workflow.set_entry_point("check_changes")

# Add the conditional edges
workflow.add_conditional_edges(
    "check_changes",
    after_check_changes,
    {
        "decide_to_commit": "decide_to_commit",
        "wait": "wait"
    }
)

workflow.add_conditional_edges(
    "decide_to_commit",
    after_decide_commit,
    {
        "generate_message": "generate_message",
        "wait": "wait"
    }
)

# Add the normal edges
workflow.add_edge("wait", "check_changes")
workflow.add_edge("generate_message", "commit_changes")
workflow.add_edge("commit_changes", "check_changes")

# Compile the graph into a runnable application
app = workflow.compile()

# ------------------------------------------------------------------
# 8. RUN THE AGENT SYSTEM
# ------------------------------------------------------------------

if __name__ == "__main__":
    
    # <-- 4. MODIFIED MAIN BLOCK -->
    
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Git Autocommiter Agent")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the agent without performing the actual git commit (for debugging)."
    )
    args = parser.parse_args()
    
    # --- Validation Checks ---
    api_key = os.environ.get("GOOGLE_API_KEY", "your-api-key")
    if not api_key or api_key == "your-api-key":
        print("="*50)
        print("❌ ERROR: Please set your 'GOOGLE_API_KEY' at the top of the script.")
        print("="*50)
    elif REPO_PATH == "path/to/your/repo":
         print("="*50)
         print("❌ ERROR: Please set the 'REPO_PATH' variable to your local repository.")
         print("="*50)
    else:
        print(f"🚀 Starting Git Autocommiter Agent...")
        
        if args.dry_run:
            print("   ⚠️  RUNNING IN DRY-RUN MODE. NO COMMITS WILL BE MADE. ⚠️")
            
        print(f"   Monitoring Repo: {REPO_PATH}")
        print(f"   Working on Feature: {FEATURE_DESCRIPTION}")
        print(f"   Check Interval: {CHECK_INTERVAL_SECONDS} seconds")
        print("="*50)
        
        # Set the initial state
        initial_state = {
            "repo_path": REPO_PATH,
            "feature_description": FEATURE_DESCRIPTION,
            "wait_interval_seconds": CHECK_INTERVAL_SECONDS,
            "changes": "",
            "commit_decision": "",
            "commit_message": "",
            "dry_run": args.dry_run, # Pass the CLI argument into the state
            "error": None
        }
        
        try:
            # The .stream() method will run the graph in a loop
            for event in app.stream(initial_state):
                # Print the current node being executed
                node = list(event.keys())[0]
                print(f"--- Entering Node: {node} ---")
                
        except KeyboardInterrupt:
            print("\n🛑 User interrupted. Shutting down agent.")
        except Exception as e:
            print(f"\n💥 An unexpected error occurred: {e}")