import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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
    repo_path: str
    messages: list
    plan: Plan | None
    deps: DepReport | None
    final: FinalTasks | None


def schema_hint(model_cls) -> str:
    return json.dumps(model_cls.model_json_schema(), indent=2)


# =========================
# Utilities
# =========================


def run(cmd: List[str], cwd: Path | None = None, limit: int = 50000) -> str:
    out = subprocess.check_output(
        cmd, cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.DEVNULL
    )
    return out[:limit]


def safe_run(cmd: List[str], cwd: Path | None = None, limit: int = 50000) -> str:
    try:
        return run(cmd, cwd, limit)
    except subprocess.CalledProcessError:
        return ""


def grep_paths(repo: str, keywords: List[str]) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for kw in keywords:
        res = safe_run(["grep", "-RIl", kw, repo])
        if res:
            files = [p.strip() for p in res.splitlines() if p.strip()]
            hits[kw] = files
    return hits


def git_cochange(repo: str, max_commits: int = 400) -> Dict[Tuple[str, str], int]:
    log = safe_run(
        ["git", "log", f"-{max_commits}", "--name-only", "--pretty=format:---"],
        cwd=Path(repo),
    )
    pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
    block: List[str] = []
    for line in log.splitlines():
        if line.strip() == "---":
            files = sorted(set(f for f in block if f and not f.startswith("---")))
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    pair_count[(files[i], files[j])] += 1
            block = []
        elif line.strip():
            block.append(line.strip())
    return pair_count


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
# Nodes
# =========================


def planner_node(state: PMState) -> PMState:
    """
    LLM-only planner. Produces 6-12 tasks covering all stories.
    Prefers seams that reduce cross-task dependency. Suggests probable files if obvious.
    """
    stories = state["jira_stories"]
    repo = state["repo_path"]

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

        Repository path (reference only): {repo}

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
    Builds a code-grounded dependency report from grep hits and git co-change.
    """
    plan = state["plan"]
    repo = state["repo_path"]
    if not plan:
        raise RuntimeError("No plan in state")

    # Keyword hits per task
    kw_files_by_task: Dict[int, List[str]] = {}
    for idx, t in enumerate(plan.tasks):
        kws = task_keywords(t.title, t.rationale, t.probable_files)
        hits = grep_paths(repo, kws)
        files = sorted({p for lst in hits.values() for p in lst})
        # include model's probable file hints
        files += [
            str(Path(repo) / pf) if not pf.startswith(str(repo)) else pf
            for pf in t.probable_files
        ]
        kw_files_by_task[idx] = sorted(set(files))

    # Co-change from git history
    co = git_cochange(repo)

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
    Merges and orders tasks using the classifier arcs.
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
# Build Graph
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
        1) Add due_date field  
        As a user I can set an ISO 8601 `due_date` on a task so I can track deadlines. Acceptance: `POST /todos` and `PUT /todos/<id>` accept `due_date` or null and persist it in SQLite. Validation: must be in the future or today; return 400 on bad format.

        2) Overdue and upcoming filters  
        As a user I can query `GET /todos?status=overdue` and `GET /todos?status=upcoming&days=7` to focus my list. Acceptance: add query params, return correct subsets; keep existing filters working together.

        3) Reminder scheduling flag  
        As a user I can mark `remind=true` to enable reminders on a task. Acceptance: store boolean; add `remind_at` derived field in responses when `due_date` exists and the reminder lead time is applied.

        4) Reminder lead-time configuration  
        As a user I can set a default reminder lead time in minutes via `PUT /settings` so reminders trigger before due dates. Acceptance: settings persisted in SQLite; tasks compute `remind_at = due_date - lead_time`.

        5) CSV export  
        As an analyst I can `GET /todos/export.csv` to download all tasks with `id,title,done,due_date,remind,remind_at`. Acceptance: correct headers and rows; dates in ISO 8601.

        6) Swagger docs and tests  
        As a developer I want Swagger updated and tests added for create/update with due dates, filters, reminder computation, and CSV export. Acceptance: `/apidocs` shows new fields; `pytest` passes.

        """

    repo_path = "../../test-repos/flask-todo-app"  # set this

    state: PMState = {
        "jira_stories": stories,
        "repo_path": repo_path,
        "messages": [],
        "plan": None,
        "deps": None,
        "final": None,
    }

    out = app.invoke(state)

    ### out["plan"] contains a Plan
    ### out["deps"] contains a DepReport
    ### out["deps"] score values:
    #           0.35–0.49 weak coupling. Just keep an eye on ordering.
    #           0.50–0.69 moderate. Prefer to sequence or add a contract/adapter first.
    #           ≥0.70 strong. Merge tasks or land a prep task (API/schema/adapter) before parallelizing.
