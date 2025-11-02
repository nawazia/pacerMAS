import json
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError
from task_creation import TaskGraph


@tool
def fs_ls(dir: str = ".", glob: str = "**/*", max_paths: int = 2000) -> List[str]:
    """List files under dir with glob. Returns relative paths."""
    print("fs_ls", dir, max_paths)
    base = Path(dir)
    paths = [str(p.relative_to(base)) for p in base.glob(glob) if p.is_file()]
    print(paths)
    return paths[:max_paths]


@tool
def fs_read(path: str, max_bytes: int = 120_000) -> str:
    """Read a text file from the repo. Returns at most max_bytes."""
    print("fs_read", path, max_bytes)
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    return p.read_text(errors="replace")[:max_bytes]


@tool
def fs_search(dir: str, query: str, max_hits: int = 200) -> Dict[str, List[int]]:
    """Search for a plain string in files under dir. Returns file -> line numbers."""
    print("fs_search", dir, query, max_hits)
    base = Path(dir)
    hits = {}
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        lines = [i for i, line in enumerate(txt.splitlines(), 1) if query in line]
        if lines:
            hits[str(p.relative_to(base))] = lines[:20]
        if len(hits) >= max_hits:
            break
    return hits


TOOLS = [fs_ls, fs_read, fs_search]


# ---------- Schemas ----------
class CommitSpec(BaseModel):
    title: str
    body: str
    files: List[str]
    per_file_changes: List[str]


class BranchSpec(BaseModel):
    name: str
    rationale: str
    tasks_included: List[str]
    parts_covered: Dict[str, List[str]] = Field(default_factory=dict)
    risk_notes: List[str] = Field(default_factory=list)
    commits: List[CommitSpec] | None = None


class BranchPlanSet(BaseModel):
    branches: List[BranchSpec]


class AssignedWork(BaseModel):
    person_to_branchplanset: Dict[str, BranchPlanSet]


# ---------- State ----------
class State(TypedDict):
    repo_path: str
    task_graph: TaskGraph  # JSON string
    messages: list  # chat history
    output: Optional[BranchPlanSet]


# ---------- LLM ----------
llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
).bind_tools(TOOLS)

SYSTEM = (
    "You are a pragmatic release engineer. You will read a task graph and propose branches and commit plans.\n"
    "Constraints:\n"
    "- Branches should be cohesive vertical or horizontal slices that minimize cross branch blocking.\n"
    "- A branch may cover multiple tasks if they are tightly related. A single task may be split when schema or API needs a prep branch.\n"
    "- You can call tools to inspect the repo: fs_ls, fs_read, fs_search but feel free to make a call and then you will be given the output but don't call each of fs_ls, fs_read, fs_search more than 10 times so 30 calls all together.\n"
    "Output ONLY a JSON object that validates against this JSON Schema:\n"
    f"{json.dumps(BranchPlanSet.model_json_schema(), indent=2)}\n"
    "Do not add extra keys. No markdown."
)


# ---------- Nodes ----------
def agent_node(state: State) -> State:
    if not state.get("messages"):
        # *** CHANGE THIS SECTION ***
        from langchain_core.messages import SystemMessage  # Ensure this is imported

        state["messages"] = [
            SystemMessage(content=SYSTEM),  # Use SystemMessage
            HumanMessage(
                content=(  # Use HumanMessage
                    "Task graph JSON is provided below. Use tools to find real files and plan commits in small, meaningful steps. "
                    "Each commit must list files and per file change summaries. "
                    f"Note you should probably beign with an fs_ls call to the root repo path. Repo path: {state['repo_path']}\n\n"
                    f"task_graph:\n{state['task_graph'].model_dump_json(indent=2)}"
                )
            ),
        ]
        # *** END OF CHANGE SECTION ***
    # Let the model decide whether to call a tool or answer
    ai = llm.invoke(state["messages"])
    state["messages"].append(ai)
    return state


# Router: if the last AI message has tool_calls, go to tools; else try to finalize
def route(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "finalize"


tool_node = ToolNode(TOOLS)


def finalize_node(state: State) -> State:
    last = state["messages"][-1]
    if isinstance(last, AIMessage):
        raw = last.content.strip()
        try:
            data = json.loads(raw)
            state["output"] = BranchPlanSet.model_validate(data)
            return state
        except (json.JSONDecodeError, ValidationError):
            # Nudge: ask for valid JSON and loop back to agent
            state["messages"].append(
                HumanMessage(  # <-- Use HumanMessage
                    content="Your output did not validate. Return ONLY valid JSON for BranchPlanSet. No prose.",
                )
            )
            return state
    # If last wasn’t AI text, loop back
    return state


# When tools run, ToolNode automatically appends ToolMessages; loop back to agent
# We also make sure the model has a default dir to avoid empty-arg calls.
def tools_wrapper(state: State) -> State:
    # Auto-fill missing args by intercepting the last AIMessage and fixing tool_calls
    ai = state["messages"][-1]
    fixed = []

    # Get the repo root as a Path object
    repo_path = Path(state["repo_path"])

    for call in ai.tool_calls:
        name = call["name"]
        args = call.get("args") or {}

        # *** START OF FIX ***

        if name in ("fs_ls", "fs_search"):
            # Get the relative dir from LLM, default to "." (repo root)
            relative_dir = args.get("dir", ".")
            # ALWAYS join it with the repo_path to create the full path
            args["dir"] = str(repo_path / relative_dir)

            # Set other defaults if missing
            if name == "fs_ls":
                args.setdefault("glob", "**/*")
                args.setdefault("max_paths", 2000)
            if name == "fs_search" and "query" not in args:
                # Keep the self-correction nudge
                args = {"dir": str(repo_path), "query": "__MISSING_QUERY__"}

        elif name == "fs_read":
            # Get the relative file path from LLM, default to README.md
            relative_path = args.get("path", "README.md")
            # ALWAYS join it with the repo_path
            args["path"] = str(repo_path / relative_path)

        # *** END OF FIX ***

        fixed.append({"name": name, "args": args, "id": call["id"]})

    ai.tool_calls = fixed  # Mutate the AIMessage with corrected, absolute paths

    # Now, execute the tools and append results (the fix from our previous conversation)
    tool_messages = tool_node.invoke(state["messages"])
    state["messages"].extend(tool_messages)

    return state


# ---------- Graph ----------
g = StateGraph(State)
g.add_node("agent", agent_node)
g.add_node("tools", tools_wrapper)
g.add_node("finalize", finalize_node)

g.set_entry_point("agent")
g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
g.add_edge("tools", "agent")  # loop back after tools
g.add_conditional_edges(
    "finalize",
    lambda s: "done" if s.get("output") else "again",
    {"done": END, "again": "agent"},
)

app = g.compile()

# --------- CLI entry ---------
if __name__ == "__main__":

    state: State = {
        "repo_path": os.getenv("GITHUB_REPO_PATH"),
        "task_graph": TaskGraph.model_validate_json(
            """{
                    "tasks": [
                        {
                        "id": "task_1",
                        "title": "Implement Weighted Edits Support",
                        "rationale": "To enhance similarity scoring by allowing certain character substitutions to count differently in edit distance calculations.",
                        "deliverables": [
                            "New API method distance_with_weights",
                            "Updated documentation page",
                            "Tests covering various weight scenarios"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        },
                        {
                        "id": "task_2",
                        "title": "Add Max-Distance Cutoff Option",
                        "rationale": "To improve performance by allowing the distance method to return early for inputs exceeding a specified threshold.",
                        "deliverables": [
                            "New methods distance(s1, s2, max_dist=N) and similarity(s1, s2, max_dist=N)",
                            "Benchmarks added",
                            "Backward compatibility ensured"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        },
                        {
                        "id": "task_3",
                        "title": "Implement Multi-Threaded Batch Computation",
                        "rationale": "To allow faster processing of large lists of strings for similarity ranking through parallel computation.",
                        "deliverables": [
                            "New batch method batch_distance(list1, list2, threads=T)",
                            "Updated documentation",
                            "New performance test case"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        },
                        {
                        "id": "task_4",
                        "title": "Output Edit Operations Path",
                        "rationale": "To provide detailed insights into the edit operations performed during distance calculations.",
                        "deliverables": [
                            "New method distance_with_ops(s1, s2)",
                            "Documentation and tests for the new method"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        },
                        {
                        "id": "task_5",
                        "title": "Support Unicode Grapheme Clusters",
                        "rationale": "To ensure multi-codepoint characters are treated as single units in distance calculations, improving correctness.",
                        "deliverables": [
                            "New optional parameter use_graphemes=True",
                            "Documentation of behavior",
                            "Tests covering grapheme vs code-point scenarios"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        },
                        {
                        "id": "task_6",
                        "title": "Introduce Pure-Python Fallback Module",
                        "rationale": "To maintain functionality on platforms where the C extension cannot compile, enhancing portability.",
                        "deliverables": [
                            "Fallback module levenshtein_py.dart",
                            "Automatic selection logic",
                            "Documentation on performance trade-off"
                        ],
                        "probable_files": [
                            "src/Levenshtein/levenshtein_cpp.pyx",
                            "src/Levenshtein/__init__.py"
                        ]
                        }
                    ],
                    "edges": [
                        {
                        "src": 1,
                        "dst": 2,
                        "reason": "Max-distance cutoff may depend on weighted edits for accurate scoring.",
                        "confidence": 0.7
                        },
                        {
                        "src": 1,
                        "dst": 3,
                        "reason": "Weighted edits may influence batch processing results.",
                        "confidence": 0.6
                        },
                        {
                        "src": 2,
                        "dst": 4,
                        "reason": "Max-distance cutoff may affect the operations path output.",
                        "confidence": 0.5
                        },
                        {
                        "src": 3,
                        "dst": 5,
                        "reason": "Batch processing may need to consider grapheme clusters for accuracy.",
                        "confidence": 0.6
                        },
                        {
                        "src": 4,
                        "dst": 5,
                        "reason": "Edit operations may involve grapheme clusters.",
                        "confidence": 0.5
                        },
                        {
                        "src": 5,
                        "dst": 6,
                        "reason": "Grapheme support may be necessary for the fallback module.",
                        "confidence": 0.4
                        }
                    ]
                    }
            """
        ),
        "messages": [],
        "plan": None,
    }
    out = app.invoke(state)
    print(json.dumps(out["output"].model_dump(), indent=2))
