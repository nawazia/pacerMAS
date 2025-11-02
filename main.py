import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

# --- Import your agent functions ---
# (Assuming they are in an 'agents' folder relative to this script)
from agents.branch_commit_creation import main as branch_commit_main
from agents.dev_profiling import main as dev_profiling_main
from agents.task_creation_old import main as task_creation_main


def save_to_json(data: dict, file_path: Path):
    """Utility function to save dictionary data to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[SUCCESS] Data successfully written to {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write data to {file_path}: {e}")


def run_git_commands(repo_path: Path):
    """Adds, commits, and pushes specified files to the git repository."""
    print("[INFO] Attempting to add, commit, and push JSON files...")

    # Define the files to be added
    files_to_add = ["plan.json", "deps.json"]
    commit_message = "Add generated agent plans and dependencies"

    # Commands to be executed
    commands = [
        ["git", "add"] + files_to_add,
        ["git", "commit", "-m", commit_message],
        ["git", "push"],
    ]

    for command in commands:
        command_str = " ".join(command)
        print(f"[INFO] Running: {command_str}")
        try:
            # Run the command from the repository's directory
            result = subprocess.run(
                command,
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.stdout:
                print(f"[GIT STDOUT] {result.stdout.strip()}")
            if result.stderr:
                print(f"[GIT STDERR] {result.stderr.strip()}")

        except FileNotFoundError:
            print(
                f"[ERROR] 'git' command not found. Please ensure Git is installed and in your PATH."
            )
            break
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Git command '{command_str}' failed with return code {e.returncode}."
            )
            print(f"[GIT STDOUT] {e.stdout.strip()}")
            print(f"[GIT STDERR] {e.stderr.strip()}")
            print("Stopping Git operations.")
            break
        except Exception as e:
            print(
                f"[ERROR] An unexpected error occurred while running Git command: {e}"
            )
            break


if __name__ == "__main__":
    # Load environment variables (e.g., from a .env file)
    load_dotenv()

    # Get the repo path from the environment
    repo_path_str = os.getenv("GITHUB_REPO_PATH")

    if not repo_path_str:
        print("[ERROR] GITHUB_REPO_PATH environment variable not set.")
        print("Please set this variable (e.g., in your .env file) and try again.")
    else:
        # Create a Path object for robust file handling
        repo_path = Path(repo_path_str).resolve()

        # --- 1. Run Task Creation Agent ---
        print("Running Task Creation Agent...")
        task_plan = task_creation_main()

        # Save the task_plan to plan.json
        plan_file_path = repo_path / "deps.json"
        print(f"Saving task plan to {plan_file_path}...")
        save_to_json(task_plan, plan_file_path)

        # --- 2. Run Development Profiling Agent ---
        print("\nRunning Development Profiling Agent...")
        dev_profiling_main()

        # --- 3. Run Branch Commit Creation Agent ---
        print("\nRunning Branch Commit Creation Agent...")
        branch_commit_output = branch_commit_main()

        # Save the branch_commit_output to deps.json
        deps_file_path = repo_path / "plan.json"
        print(f"Saving branch/commit output to {deps_file_path}...")
        save_to_json(branch_commit_output, deps_file_path)

        # --- 4. Add, Commit, and Push generated files ---
        print("\nRunning Git commands to save plans...")
        run_git_commands(repo_path)

        print("\n--- All agents finished ---")
