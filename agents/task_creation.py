import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TypedDict
import base64

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# GitHub API specific imports
from github import Github, Repository, Auth
from github.ContentFile import ContentFile

# --- Configuration ---
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") # Load the token globally

# =========================
# Models and State
# =========================

class TaskCandidate(BaseModel):
    id: str
    title: str
    rationale: str
    deliverables: List[str] = Field(default_factory=list)
    probable_files: List[str] = Field(default_factory=list)


class Plan(BaseModel):
    tasks: List[TaskCandidate]


class DepArc(BaseModel):
    a_idx: int
    b_idx: int
    reason: str
    score: float


class DepReport(BaseModel):
    arcs: List[DepArc]


class FinalTasks(BaseModel):
    tasks: List[str]  # ordered and possibly merged


class PMState(TypedDict):
    jira_stories: str
    repo_url: str # CHANGED: Uses URL instead of local path
    messages: list
    plan: Plan | None
    deps: DepReport | None
    final: FinalTasks | None


def schema_hint(model_cls) -> str:
    return json.dumps(model_cls.model_json_schema(), indent=2)


# =========================
# GitHub Agent Utilities (Replaces local subprocess/git/grep)
# =========================

class GitHubAgent:
    def __init__(self, repo_url: str, token: Optional[str] = None):
        if not token:
            raise ValueError("GITHUB_TOKEN is required for GitHub API access.")
        
        # Authenticate and get the repository object
        g = Github(auth=Auth.Token(token))
        
        # Parse owner/repo_name from URL (e.g., https://github.com/owner/repo)
        path_parts = repo_url.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL format: {repo_url}")
            
        repo_path = f"{path_parts[-2]}/{path_parts[-1]}"
        self.repo: Repository = g.get_repo(repo_path)
        self.repo_name = repo_path

    def get_repo_files(self) -> List[str]:
        """Recursively get a list of all file paths in the main branch."""
        file_paths = []
        contents = self.repo.get_contents("")
        
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                # For large repos, you might hit rate limits or memory issues.
                # A robust solution should use the Git Tree API.
                try:
                    contents.extend(self.repo.get_contents(file_content.path))
                except Exception as e:
                    print(f"Skipping directory {file_content.path} due to error: {e}")
            else:
                file_paths.append(file_content.path)
        return file_paths

    def grep_paths(self, keywords: List[str], all_files: List[str]) -> Dict[str, List[str]]:
        """
        Simulates `grep -RIl` by checking file contents for keywords.
        """
        hits: Dict[str, List[str]] = defaultdict(list)
        
        # Create a single regex-like pattern string (simple keyword match for this demo)
        keywords_lower = [kw.lower() for kw in keywords]
        
        for file_path in all_files:
            try:
                # Get the file content
                content_file: ContentFile = self.repo.get_contents(file_path)
                
                # Content is base64 encoded for files > 1KB
                content_bytes = content_file.content.encode('utf-8')
                content = base64.b64decode(content_bytes).decode('utf-8')
                content_lower = content.lower()
                
                # Check for hits and store the file path under the keyword
                for kw in keywords_lower:
                    if kw in content_lower:
                        hits[kw].append(file_path)
                        # Optimization: stop checking this file after the first hit
                        break 
                        
            except Exception:
                # Skip binary files or API errors (e.g., encoding/decoding)
                continue
                
        # The original function required a Dict[str, List[str]] where keys were keywords
        # and values were files matching that keyword.
        return {k: list(set(v)) for k, v in hits.items()}

    def git_cochange(self, max_commits: int = 400) -> Dict[Tuple[str, str], int]:
        """
        Simulates `git log --name-only` to find file co-change frequency.
        (Updated to remove 'per_page' which causes an error in some PyGithub versions)
        """
        pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
        
        # --- FIX: Remove 'per_page' argument ---
        # Fetch the last N commits using slicing on the resulting iterable.
        # get_commits() retrieves commits; slicing limits the number we process.
        commits = self.repo.get_commits()[:max_commits] 
        
        for commit in commits:
            # Get files modified in the commit
            try:
                # commit.files is an expensive API call; use sparingly.
                files = [f.filename for f in commit.files]
            except Exception:
                # Handle cases where commit data is restricted
                continue
                
            files = sorted(set(files))
            
            # Count co-changes
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    # Note: We use tuple(sorted(...)) to ensure the key is consistent regardless of (fi, fj) order
                    pair_count[tuple(sorted((files[i], files[j])))] += 1
                    
        return pair_count


# =========================
# General Utilities (Unchanged)
# =========================

def task_keywords(title: str, rationale: str, probable_files: List[str]) -> List[str]:
    base = f"{title} {rationale} " + " ".join(probable_files)
    toks = [w.strip(".,:;()[]{}").lower() for w in base.split()]
    return [w for w in toks if len(w) >= 4][:10]


# =========================
# LLMs
# =========================

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)

planner_llm = llm.with_structured_output(Plan)
merger_llm = llm.with_structured_output(FinalTasks)

# =========================
# Nodes (Modified for GitHub Agent)
# =========================


def planner_node(state: PMState) -> PMState:
    """
    LLM-only planner. Uses repo_url for context.
    """
    stories = state["jira_stories"]
    repo_url = state["repo_url"] # Changed to repo_url

    system = SystemMessage(
        content=(
            "You are a senior project planner. Create 6-12 implementation tasks that together cover all stories. "
            "Minimize cross-task dependencies by choosing clean boundaries like APIs, schema migrations, adapters, or flags. "
            "Each task must include a concise title, a short rationale, clear deliverables, and probable files if you can infer them."
        )
    )
    human = HumanMessage(
        content=f"""
            JIRA STORIES:
        {stories}

        Repository URL (reference only): {repo_url}

        Output contract:
        You MUST return a JSON object that validates against this JSON Schema for Plan:
        {schema_hint(Plan)}

        Guidelines:
        - Create 6 to 12 tasks total.
        - Each TaskCandidate.title is a short string.
        - TaskCandidate.rationale is a short string explaining the boundary.
        - TaskCandidate.deliverables is a list of strings.
        - TaskCandidate.probable_files is a list of relative file paths (strings). If unknown, use [].
        - Choose seams that minimize cross-task dependency (APIs, schema migrations, adapters, flags).
        Return ONLY the JSON for Plan. No prose.
        """
    )

    plan: Plan = planner_llm.invoke([system, human])
    state["plan"] = plan
    state["messages"] = state.get("messages", []) + [system, human]
    return state


def classifier_node(state: PMState) -> PMState:
    """
    Builds a code-grounded dependency report using the GitHub API agent.
    """
    plan = state["plan"]
    repo_url = state["repo_url"]
    if not plan:
        raise RuntimeError("No plan in state")
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set in environment.")

    # 1. Initialize the GitHub Agent
    agent = GitHubAgent(repo_url, GITHUB_TOKEN)
    
    # 2. Get all file paths in the repo
    all_files = agent.get_repo_files()

    # Keyword hits per task
    kw_files_by_task: Dict[int, List[str]] = {}
    for idx, t in enumerate(plan.tasks):
        kws = task_keywords(t.title, t.rationale, t.probable_files)
        # Use the agent's grep_paths
        hits = agent.grep_paths(kws, all_files) 
        files = sorted({p for lst in hits.values() for p in lst})
        
        # include model's probable file hints (which should be relative paths)
        files.extend(t.probable_files) 
        kw_files_by_task[idx] = sorted(set(files))

    # Co-change from git history
    co = agent.git_cochange()

    arcs: List[DepArc] = []
    n = len(plan.tasks)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ti = plan.tasks[i]
            tj = plan.tasks[j]

            files_i = set(kw_files_by_task.get(i, []))
            files_j = set(kw_files_by_task.get(j, []))

            shared = files_i & files_j
            shared_score = 0.6 if shared else 0.0

            hist = 0
            for fi in files_i:
                for fj in files_j:
                    key = tuple(sorted((fi, fj)))
                    hist += co.get(key, 0)
            hist_score = min(hist / 10.0, 0.6)

            # Text overlap as weak proxy
            ti_k = set(task_keywords(ti.title, ti.rationale, ti.probable_files))
            tj_k = set(task_keywords(tj.title, tj.rationale, tj.probable_files))
            text_score = min(len(ti_k & tj_k) / 5.0, 0.3)

            score = min(shared_score + hist_score + text_score, 1.0)
            if score >= 0.35:
                reason = json.dumps(
                    {
                        "shared": list(shared)[:3],
                        "hist_pairs": hist,
                        "kw_overlap": list(ti_k & tj_k)[:5],
                    }
                )
                arcs.append(DepArc(a_idx=i, b_idx=j, reason=reason, score=score))

    state["deps"] = DepReport(arcs=arcs)
    return state


def merger_node(state: PMState) -> PMState:
    """
    Merges and orders tasks using the classifier arcs. (Unchanged)
    """
    plan = state["plan"]
    deps = state["deps"]
    if not plan or not deps:
        raise RuntimeError("Missing plan or deps")

    strong_pairs = {(a.a_idx, a.b_idx) for a in deps.arcs if a.score >= 0.7}

    system = SystemMessage(
        content=(
            "You are a project manager. Merge strongly coupled tasks and output an ordered list that minimizes blockers. "
            "Prefer boundary tasks first, then consumers. Keep near original task count ±2 and ensure coverage of all stories."
        )
    )

    human = HumanMessage(
        content=f"""
        Context:
        TASKS (indexed):
        {json.dumps([t.model_dump() for t in plan.tasks], indent=2)}

        CLASSIFIER ARCS (a->b with score):
        {json.dumps([a.model_dump() for a in deps.arcs], indent=2)}

        STRONG PAIRS TO MERGE:
        {json.dumps(list(map(list, strong_pairs)), indent=2)}

        Goal:
        - Merge strongly coupled tasks when decoupling is not clean.
        - Order tasks to minimize blockers. Put boundary tasks (APIs/schemas/adapters/flags) first.
        - Keep total count near original ±2 and ensure coverage of all stories.

        Output contract:
        Return ONLY a JSON object that validates against this JSON Schema for FinalTasks:
        {schema_hint(FinalTasks)}

        Notes:
        - FinalTasks.tasks is an ordered list of concise task titles (strings).
        - No prose, no markdown, just the JSON.
        """
    )

    final: FinalTasks = merger_llm.invoke([system, human])
    state["final"] = final
    state["messages"].extend([system, human])
    return state


# =========================
# Build Graph (Unchanged)
# =========================

graph = StateGraph(PMState)
graph.add_node("planner", planner_node)
graph.add_node("classifier", classifier_node)
graph.add_node("merger", merger_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "classifier")
graph.add_edge("classifier", "merger")
graph.add_edge("merger", END)

app = graph.compile()

# =========================
# Demo
# =========================

if __name__ == "__main__":
    stories = """
        1) As a developer I want “weighted edits” support so that certain character substitutions (e.g., accented vs non-accented, case differences, punctuation) count for less/more than 1 in the edit distance, enabling more flexible similarity scoring.
        
        Acceptance criteria: new API/method distance_with_weights(s1, s2, weights_map) added; documentation page updated; tests covering various weight scenarios.
        

        2) As a developer I want a “max-distance cutoff” option for the core distance method so that when two strings differ by more than a given threshold the method returns None or a sentinel, improving performance for large input sets.
        
        Acceptance criteria: methods distance(s1, s2, max_dist=N) or similarity(s1, s2, max_dist=N) introduced; bench-marks added; existing API backwards compatible.
        

        3) As a user I want multi-threaded / parallel batch computation of distances so I can feed large lists of strings and get vectorized results faster for bulk similarity ranking.
        
        Acceptance criteria: new batch method batch_distance(list1, list2, threads=T) or similar; documentation updated; new performance test case; API aligned with library style.
        

        4) As an API user I want “edit operations path” output (in addition to the numeric distance) so I can inspect exactly which insertions, deletions, substitutions were computed, enabling downstream transformation logic.
        
        Acceptance criteria: method distance_with_ops(s1, s2) returns struct {distance: int, ops: [ {type: “insert”|“delete”|“substitute”, pos1: int, pos2: int, char1: str, char2: str} ] }; docs + tests.
        

        5) As a maintainer I want the library to support Unicode grapheme clusters rather than naive code-points so that multi-codepoint characters (emojis, accented letters, combined sequences) are treated as a single unit in distance calculations, improving correctness for modern text.
        
        Acceptance criteria: new optional parameter use_graphemes=True; documentation of behavior; tests covering grapheme vs code-point scenarios.
        

        6) As a Python user I want a pure-Python fallback module so that on platforms where the C extension cannot compile I still get functional albeit slower behavior, improving portability.
        
        Acceptance criteria: fallback module levenshtein_py.dart (or similar) introduced; automatic selection if C extension load fails; docs note performance trade-off; tests validate equivalence with C version on small input.
    """

    # --- CHANGED: Use a public GitHub repo URL for the demo ---
    repo_url = "https://github.com/rapidfuzz/Levenshtein" # Replace with your target repo

    state: PMState = {
        "jira_stories": stories,
        "repo_url": repo_url,
        "messages": [],
        "plan": None,
        "deps": None,
        "final": None,
    }

    print(f"--- Running LangGraph PM on repo: {repo_url} ---")
    
    try:
        out = app.invoke(state)
        print("\n--- Final Output ---")
        print(json.dumps(out["plan"].model_dump(), indent=2))

        with open("plan.json", "w") as f:
            json.dump(out["plan"].model_dump(), f, indent=2)
        
        # Optional: Print the full dependency report
        # print("\n--- Dependency Report ---")
        # print(json.dumps(out["deps"].model_dump(), indent=2))
        
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"An error occurred: {e}")
        print("Ensure OPENROUTER_API_KEY and GITHUB_TOKEN are set correctly.")
