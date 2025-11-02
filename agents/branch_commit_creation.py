import json
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError

from agents.slack_writer import run_agent

load_dotenv()


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


@tool
def fs_ls(dir: str = ".", glob: str = "**/*", max_paths: int = 2000) -> List[str]:
    """List files under dir with glob. Returns relative paths."""
    print("fs_ls", dir, max_paths)
    base = Path(dir)
    paths = [str(p.relative_to(base)) for p in base.glob(glob) if p.is_file()]
    print(paths)
    return paths[:max_paths]


TOOLS = [fs_ls]


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
    """
    The final work assignment plan.
    Maps a developer's handle (e.g., '@maxbachmann') to a BranchPlanSet
    containing all branches (and their commits) assigned to them.
    """

    person_to_branchplanset: Dict[str, BranchPlanSet]


# ---------- State ----------
class State(TypedDict):
    repo_path: str
    task_graph: TaskGraph  # JSON string
    messages: list  # chat history
    output: Optional[BranchPlanSet]


# ---------- LLM ----------
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
).bind_tools(TOOLS)

SYSTEM = (
    "You are a pragmatic release engineer. You will read a task graph and propose branches and commit plans.\n"
    "Constraints:\n"
    "- Branches should be cohesive vertical or horizontal slices that minimize cross branch blocking.\n"
    "- A branch may cover multiple tasks if they are tightly related. A single task may be split when schema or API needs a prep branch.\n"
    "- You can call tools to inspect the repo: fs_ls but feel free to make a call and then you will be given the output but don't call fs_ls more than 10 times.\n"
    "Output ONLY a JSON object that validates against this JSON Schema:\n"
    f"{json.dumps(BranchPlanSet.model_json_schema(), indent=2)}\n"
    "Do not add extra keys. No markdown."
)


# ---------- Nodes ----------
def agent_node(state: State) -> State:
    print("AGENT CALL")
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
    print("FINLIZE NODE")
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
    print("TOOLS WRAPPER    ")
    # Auto-fill missing args by intercepting the last AIMessage and fixing tool_calls
    ai = state["messages"][-1]
    fixed = []

    # Get the repo root as a Path object and RESOLVE it to an absolute path
    repo_path = Path(state["repo_path"]).resolve()
    repo_path_str = str(repo_path)  # Get the string of the absolute path

    for call in ai.tool_calls:
        name = call["name"]
        args = call.get("args") or {}

        if name in ("fs_ls"):
            # Get the dir path from the LLM
            llm_path_str = args.get("dir", ".")

            # --- START OF FIX ---
            # Resolve the LLM's path to an absolute path
            # This handles '.', '..', and absolute paths correctly
            llm_path_resolved = Path(llm_path_str).resolve()
            llm_path_resolved_str = str(llm_path_resolved)

            # Check if the LLM's resolved path is *already inside* the repo's resolved path
            if llm_path_resolved_str.startswith(repo_path_str):
                # If YES: The LLM is re-using a full path. Use it as-is.
                args["dir"] = llm_path_resolved_str
            else:
                # If NO: Assume it's a new path relative to the repo root.
                # Join it with the absolute repo_path.
                args["dir"] = str(repo_path / llm_path_str)
            # --- END OF FIX ---

            if name == "fs_ls":
                args.setdefault("glob", "**/*")
                args.setdefault("max_paths", 2000)

        fixed.append({"name": name, "args": args, "id": call["id"]})

    ai.tool_calls = fixed  # Mutate the AIMessage with corrected, absolute paths

    # Execute tools and append results
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
def main():

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

    developer_profiles = """
    ---

## 👤 Developer: @Joerki

Based on the provided data, Joerki appears to have expertise in licensing and legal aspects within software development. They made a single commit to the repository, where they significantly modified the 'LICENSE' file by adding 13 lines and deleting 59 lines. This suggests a focus on maintaining and updating licensing terms, which is crucial for ensuring compliance and intellectual property protection in software projects. Joerki's indicated contribution aligns more with a role related to legal compliance or copyright management rather than specific programming languages or frameworks, indicating a valuable focus on the regulatory and governance aspects of software development.

---

## 👤 Developer: @antoinetavant

Based on the provided data, 'antoinetavant' has made a single commit touching the 'docs' directory, specifically the 'index.rst' file by adding 1 line of code without any deletions. This indicates their focus on documentation-related tasks. Their expertise likely lies in maintaining project documentation, ensuring clarity and accuracy in describing project features and functionalities. Being involved in documenting project information is crucial for enhancing project understanding and developer collaboration. 'antoinetavant' may assume a Documentation role within the development process, contributing to the readability and accessibility of project resources for both internal and external audiences. The preferred language or framework cannot be determined from the provided data, as the contribution is in the form of documentation.

---

## 👤 Developer: @LecrisUT

Based on the contributions by 'LecrisUT' to the GitHub repository, they have made a total of 2 commits and touched files related to 'pyproject.toml'. The first commit involved making 2 additions and 2 deletions to the 'pyproject.toml' file, while the second commit added 2 lines without any deletions to the same file. From this data, it can be inferred that 'LecrisUT' is proficient in Python and likely specializes in Python project configurations and dependencies management. Their focus seems to be on backend development or project setup/configuration tasks rather than frontend development or design. Given the nature of the changes in the 'pyproject.toml' file, 'LecrisUT' could be categorized as a Backend Developer or a DevOps specialist with expertise in Python project setups and configurations.

---

## 👤 Developer: @dependabot[bot]

Based on the contributions of 'dependabot[bot]' in the repository, they have made a single commit that involved touching the '.github/workflows/pythonbuild.yml' file. This file is related to GitHub workflows and specifically for Python builds. The additions and deletions are equal at 2 each, indicating a balanced modification. This suggests that their core expertise lies in managing and automating workflows, particularly in the context of building Python applications. Therefore, 'dependabot[bot]' likely specializes in DevOps tasks related to CI/CD pipelines and automation, with a focus on Python projects.

---

## 👤 Developer: @guyrosin

Based on the provided contribution data for 'guyrosin', it appears that they have expertise in Python programming language. The developer has made modifications to the 'levenshtein_cpp.pyx' file within the 'src/Levenshtein' directory. The equal number of additions and deletions implies that these modifications were likely for refactoring or optimizing existing code rather than adding new features. The use of '.pyx' extension suggests involvement in Cython, a tool for blending Python with C/C++ to improve performance. As the changes are in a specific directory related to algorithms (Levenshtein), 'guyrosin' may specialize in algorithm design and optimization. The provided data does not indicate a specific role, but their focus on low-level optimizations and algorithmic efficiency hints at a Back-end Developer or Software Engineer role with a specialization in performance tuning and algorithm implementation.

---

## 👤 Developer: @maxbachmann

Based on the contributions of 'maxbachmann' to the GitHub repository, it is evident that their core expertise lies in Python development. They have made a significant number of commits and touched a wide variety of files related to a Python library. Their focus seems to be on backend development, specifically working on a library named 'Levenshtein'. Additionally, 'maxbachmann' has expertise in working with CMakeLists and C++ files given the additions and deletions made to these types of files. Their role can be identified as a Backend Developer with specialization in Python and C++ development. The preferred language/framework is Python, and they have a strong emphasis on maintaining and updating documentation as seen in the numerous changes made to 'HISTORY.md' and 'docs' files."""

    out = app.invoke(state)
    plan_json = json.dumps(out["output"].model_dump(), indent=2)
    print("Plan JSON generated")

    # Create the user prompt containing the two inputs
    human_prompt = f"""
    ## 1. Developer Profiles

    {developer_profiles}

    ---

    ## 2. Branch & Commit Plan

    {plan_json}
    """

    SYSTEM_PROMPT = f"""
        You are an expert technical project manager. Your task is to assign planned work to developers.

        You will receive two inputs:
        1.  **Developer Profiles:** A list of developers and their inferred expertise.
        2.  **Branch & Commit Plan:** A `BranchPlanSet` JSON object detailing branches and their constituent commits.

        **Your Goal:**
        Assign each **entire branch** to the single developer best suited to implement it.

        **Assignment Constraints:**
        1.  **Match Expertise:** Base your assignment on the developer's expertise (from their profile) and the files/changes listed in the commits (e.g., `.pyx` files, `pyproject.toml`, `docs/`).
        2.  **Minimize Blocking:** Ideally, a single developer must own all commits on a branch. This helps to prevent merge conflicts, however, if there isn't enough work for this to happen, assign continguous sections of commits in a branch to each developer to minimize blocking.
        3.  **Distribute Work:** Distribute work among all developers as evenly as possible, so that each developer has roughly the same number of commits.
        4.  **Assign All Branches:** There could be multiple developers working on a branch, but only if needed. Ideally, if there are enough branches, each developer is assigned their own branches.
        5.  **Filter Developers:** Only assign coding/documentation tasks to developers with relevant expertise. Do not assign work to bots (like '@dependabot[bot]') or purely legal contributors (like '@Joerki').

        Rename the branches to standardized names like `feature/weighted-edits`, `feature/max-distance-cutoff`, etc.
        **Output Format:**
        You MUST output ONLY a valid JSON object that conforms to the `AssignedWork` schema. Do not add any other text, markdown, or explanation.

        **Output Schemas:**

        ```json
        {json.dumps(AssignedWork.model_json_schema(), indent=2)}"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=human_prompt),
    ]

    llm2 = ChatOpenAI(
        model="anthropic/claude-3.7-sonnet",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )

    # Invoke the model
    response = llm2.invoke(messages)

    try:
        # Parse the JSON response
        output_data = json.loads(response.content.strip())

        # Validate against the Pydantic model
        assigned_work = AssignedWork.model_validate(output_data)
        print("✅ Work assignment validated successfully.")

        run_agent(
            "Send on slack to #all-agentsverse-hackathon: BRANCHES AND COMMITS ASSIGNED"
        )
        # run_agent(
        #     "Format and send the following to #all-agentsverse-hackathon:"
        #     + json.dumps(assigned_work.model_dump_json())
        # )
        return assigned_work.model_dump()
    except json.JSONDecodeError:
        print(
            f"❌ Error: LLM did not return valid JSON.\nRaw output: {response.content}"
        )
        raise
    except Exception as e:
        print(f"❌ Error: Failed to validate LLM output.\nError: {e}")
        print(f"Raw output: {response.content}")
        raise
