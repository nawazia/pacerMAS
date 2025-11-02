# SINGLE-AGENT GRAPH OUTPUT (tasks + dependency edges)
# pip install langgraph langchain-openai pydantic python-dotenv requests

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GITHUB_REPO_PATH = os.getenv("GITHUB_REPO_PATH")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


# ---------- Schemas ----------
class TaskNode(BaseModel):
    id: str  # stable short id, e.g. T1, T2
    title: str  # concise name
    rationale: str  # why this is a clean boundary
    deliverables: List[str] = Field(default_factory=list)
    probable_files: List[str] = Field(default_factory=list)  # relative repo paths


class GraphEdge(BaseModel):
    src: int  # index in tasks list
    dst: int  # index in tasks list
    reason: str  # brief why dependency exists
    confidence: float = Field(ge=0, le=1)


class TaskGraph(BaseModel):
    tasks: List[TaskNode]
    edges: List[GraphEdge]  # directed: src -> dst must come before


class AgentState(TypedDict):
    jira_stories: str
    repo_url: str
    messages: list
    plan: TaskGraph | None


def schema_hint(model_cls) -> str:
    return json.dumps(model_cls.model_json_schema(), indent=2)


# ---------- GitHub helpers ----------
def _parse_repo_slug(url_or_slug: str) -> Tuple[str, str]:
    s = url_or_slug.rstrip("/")
    if "github.com" in s:
        parts = s.split("/")
        return parts[-2], parts[-1]
    if "/" in s and " " not in s:
        owner, repo = s.split("/", 1)
        return owner, repo
    raise ValueError(f"Cannot parse GitHub repo from: {url_or_slug}")


def _run(cmd: list[str], cwd: str | None = None, limit: int = 200000) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True, stderr=subprocess.DEVNULL)[
        :limit
    ]


def _rg_available() -> bool:
    try:
        subprocess.check_output(
            ["rg", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def _github_code_search(
    owner: str, repo: str, keyword: str, per_page: int = 50
) -> List[str]:
    """
    LOCAL VERSION: `repo` is a filesystem path to the repo.
    Prefer ripgrep; fall back to grep. Returns relative file paths.
    """
    base = Path(repo).resolve()
    if _rg_available():
        try:
            out = _run(["rg", "-l", keyword, str(base)])
            paths = [
                str(Path(p.strip()).resolve().relative_to(base))
                for p in out.splitlines()
                if p.strip()
            ]
            return paths[:per_page]
        except subprocess.CalledProcessError:
            return []
    # grep fallback
    try:
        out = _run(["grep", "-RIl", keyword, str(base)])
        paths = [
            str(Path(p.strip()).resolve().relative_to(base))
            for p in out.splitlines()
            if p.strip()
        ]
        return paths[:per_page]
    except subprocess.CalledProcessError:
        return []


def _github_commit_shas(owner: str, repo: str, max_commits: int = 200) -> List[str]:
    """
    LOCAL VERSION: `repo` is a filesystem path to the repo.
    Returns the latest commit SHAs.
    """
    try:
        out = _run(["git", "log", f"-{max_commits}", "--pretty=%H"], cwd=repo)
        shas = [ln.strip() for ln in out.splitlines() if ln.strip()]
        return shas[:max_commits]
    except subprocess.CalledProcessError:
        return []


def _github_commit_files(owner: str, repo: str, sha: str) -> List[str]:
    """
    LOCAL VERSION: `repo` is a filesystem path to the repo.
    Returns file paths (relative to repo) changed in a given commit.
    """
    try:
        out = _run(["git", "show", "--name-only", "--pretty=format:", sha], cwd=repo)
        files = []
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # ensure path is within repo and normalize to relative
            p = Path(repo, ln).resolve()
            try:
                files.append(str(p.relative_to(Path(repo).resolve())))
            except ValueError:
                # skip files outside repo root (shouldn't happen)
                continue
        return files
    except subprocess.CalledProcessError:
        return []


def _code_search_path(base: Path, keyword: str, limit: int) -> List[str]:
    globs_exclude = [
        "--glob",
        "!.git/**",
        "--glob",
        "!**/docs/**",
        "--glob",
        "!**/HISTORY.md",
        "--glob",
        "!**/CHANGELOG*",
        "--glob",
        "!**/*.md",
        "--glob",
        "!**/tests/**",
    ]
    if _rg_available():
        try:
            out = _run(["rg", "-l", "-S", "-w", keyword, str(base), *globs_exclude])
            paths = [
                str(Path(p).resolve().relative_to(base))
                for p in out.splitlines()
                if p.strip()
            ]
            return paths[:limit]
        except subprocess.CalledProcessError:
            return []
    # grep fallback
    try:
        out = _run(["grep", "-RIl", keyword, str(base)])
        paths = [
            str(Path(p).resolve().relative_to(base))
            for p in out.splitlines()
            if p.strip()
        ]
        # cheap exclude
        paths = [
            p
            for p in paths
            if not (p.endswith(".md") or p.startswith("tests/") or "/docs/" in p)
        ]
        return paths[:limit]
    except subprocess.CalledProcessError:
        return []


# ---------- Tools exposed to the agent ----------
@tool
def gh_search_keywords(
    repo_path: str, keywords: List[str], per_keyword: int = 10
) -> Dict[str, List[str]]:
    """
    Search a local Git repository for files related to specified keywords.

    Args:
        repo_path: Filesystem path to the local Git repo root.
        keywords: List of search keywords to match against code.
        per_keyword: Maximum number of file paths to return per keyword.

    Returns:
        A dictionary mapping each keyword to a list of relative file paths
        where that keyword appears. Only real source files are returned when
        possible (not docs or tests). Useful for letting the agent infer
        which files a task will likely affect.
    """
    print("keywords")
    base = Path(repo_path).resolve()
    out = {}
    for kw in keywords[:12]:
        paths = _code_search_path(base, kw, per_keyword)
        if paths:
            out[kw] = paths
    print(out)
    return out


@tool
def gh_cochange(limit_commits: int = 200) -> Dict[str, int]:
    """Return simple co-change counts: 'fileA|fileB' -> count from recent commits."""
    owner, repo = _parse_repo_slug(GITHUB_REPO_PATH)
    shas = _github_commit_shas(owner, repo, max_commits=limit_commits)
    counts: Dict[str, int] = {}
    for sha in shas:
        files = sorted(set(_github_commit_files(owner, repo, sha)))
        n = len(files)
        if n == 0 or n > 200:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{files[i]}|{files[j]}"
                counts[key] = counts.get(key, 0) + 1
    return counts


# ---------- LLM with tools ----------
base_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)
agent_llm = base_llm.bind_tools([gh_search_keywords, gh_cochange])


def invoke_single_agent(jira_stories: str, repo_url: str) -> TaskGraph:
    sys = SystemMessage(
        content=(
            "You are a single planning agent. You can call tools to inspect the repo. "
            "Goal: produce a task graph that covers all JIRA stories while minimizing cross-task dependencies. "
            "Use gh_search_keywords to see likely files per concept and gh_cochange to spot coupled files. Note that the goal is that you give probable_files for each task which should be as comprehensive as possible and include all existing files that completing the task will involve. The edges should model dependency with a score per edge saying how strong the dependency is"
            "When you are done, OUTPUT ONLY a JSON object that validates against this JSON Schema:\n"
            f"{schema_hint(TaskGraph)}\n"
            "Rules:\n"
            "- 6 to 12 tasks.\n"
            "- Each task has id, title, rationale, deliverables[], probable_files[].\n"
            "- Edges are directed src->dst meaning src should be done before dst.\n"
            "- confidence is 0..1.\n"
            "- No markdown, no extra keys, valid JSON only."
        )
    )
    human = HumanMessage(
        content=f"JIRA STORIES:\n{jira_stories}\n\nRepo: {repo_url}\n"
        "Plan boundary-first tasks (APIs/schemas/adapters/flags) to reduce blocking."
    )

    messages = [sys, human]

    for i in range(12):
        ai: AIMessage = agent_llm.invoke(messages)
        print("AI Message received")
        if getattr(ai, "tool_calls", None):
            print(f"Tool call number {i} detected")
            messages.append(ai)
            for tc in ai.tool_calls:
                if tc["name"] == "gh_search_keywords":
                    # ensure tc["args"] includes {"repo_path": repo_path, "keywords":[...]}
                    result = gh_search_keywords.invoke(tc["args"])
                elif tc["name"] == "gh_cochange":
                    result = gh_cochange.invoke(tc["args"])
                else:
                    result = {"error": "unknown_tool"}
                messages.append(
                    ToolMessage(
                        content=json.dumps(result)[:6000], tool_call_id=tc["id"]
                    )
                )
            continue

        # Try to parse final graph
        raw = ai.content.strip()
        try:
            data = json.loads(raw)
            return TaskGraph.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            messages.append(AIMessage(content=ai.content))
            messages.append(
                HumanMessage(
                    content="Output did not validate. Return ONLY valid JSON for TaskGraph."
                )
            )
            continue

    raise RuntimeError("Agent finished without a valid TaskGraph.")


# ---------- LangGraph: single node ----------
def agent_node(state: AgentState) -> AgentState:
    graph = invoke_single_agent(state["jira_stories"], state["repo_url"])
    state["plan"] = graph
    return state


graph = StateGraph(AgentState)
graph.add_node("planner_agent", agent_node)
graph.add_edge(START, "planner_agent")
graph.add_edge("planner_agent", END)
app = graph.compile()

# ---------- Demo ----------
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

    state: AgentState = {
        "jira_stories": stories,
        "repo_url": GITHUB_REPO_PATH,
        "messages": [],
        "plan": None,
    }
    out = app.invoke(state)
    print(json.dumps(out["plan"].model_dump(), indent=2))
