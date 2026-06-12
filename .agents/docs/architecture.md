# Agentic ML Loop Architecture

Agentic ML Loop runs bounded ML research cycles against a local experiment
directory. Each cycle builds a prompt from the experiment files, invokes a
configured agent CLI, validates what the agent produced, and folds the result
into persisted state. The experiment directory is the single source of truth;
the framework never holds long-lived state of its own.

## System overview

Three views: the **components** and how they connect, the **one-cycle** flow,
and the **loop lifecycle** that decides when to stop.

### Components & boundaries

A single pipeline in cycle order: the CLI drives the supervisor, which builds a
prompt, invokes the external agent, validates the attempt, and runs post-cycle
hooks. Framework code (blue) reads and writes the experiment directory (green)
but never imports experiment code (grey) — runners are the only bridge.

```mermaid
flowchart LR
    CLI(["python -m loop<br/>start · resume · status · bench"])

    subgraph fw ["Framework — loop/ + lib/*.py"]
        direction LR
        Core["core.py<br/>supervisor"]
        Prompts["prompts.py<br/>build prompt"]
        Invoke["invoke.py<br/>spawn subprocess"]
        Attempt["cycle_attempt.py<br/>validate attempt"]
        Hooks["hooks.py<br/>progress · referee · learnings"]
    end

    Agent["External agent CLI<br/>claude · codex · cursor · custom"]

    subgraph store ["experiments/&lt;id&gt;/ — persisted state"]
        direction TB
        SpecJournal["experiment.md · research_sources.md<br/>research_journal.md · results.json"]
        LoopState["loop_state.json · status.md · cycles/<br/>outputs/ · work/ · scripts/"]
    end

    subgraph expcode ["Experiment code — reached only via runners"]
        direction LR
        Runners["runners/demo_*_runner.py"] --> Lib["lib/demo_*/<br/>data.py · modeling.py"]
    end

    CLI --> Core
    Core --> Prompts --> Invoke --> Agent --> Attempt --> Hooks
    Core -. reads .-> SpecJournal
    Agent -. writes .-> SpecJournal
    Hooks -- updates --> LoopState
    Runners -. seed + run .-> store

    classDef fwNode fill:#e8f0fe,stroke:#4285f4,color:#202124;
    classDef agentNode fill:#fef7e0,stroke:#f9ab00,color:#202124;
    classDef stateNode fill:#e6f4ea,stroke:#34a853,color:#202124;
    classDef codeNode fill:#f1f3f4,stroke:#9aa0a6,color:#202124;
    class Core,Prompts,Invoke,Attempt,Hooks fwNode;
    class Agent agentNode;
    class SpecJournal,LoopState stateNode;
    class Runners,Lib codeNode;
```

### One cycle

`run_cycle` snapshots the experiment, builds the prompt, and retries one runner
attempt up to three times. Each attempt is checked against the cycle contract;
a failing attempt rolls back and retries, an exhausted cycle fails, and a clean
attempt runs the post-cycle hooks before persisting state.

```mermaid
flowchart TB
    Start(["run_cycle"]) --> Snap["snapshot baselines<br/>journal hash + experiment.md"]
    Snap --> Prompt["build prompt<br/>program.md + guidelines + state"]
    Prompt --> Loop

    subgraph Loop ["single attempt (up to 3)"]
        direction TB
        Restore["restore baselines"] --> Run["invoke runner on stdin<br/>capture stdout / stream-json"]
        Run --> Extract["extract assistant text<br/>+ completion marker"]
        Extract --> Validate["validate_experiment<br/>+ contract check"]
    end

    Loop --> Clean{contract<br/>clean?}
    Clean -- "no · attempts left" --> Restore
    Clean -- "no · exhausted" --> Fail["roll back<br/>result = failed"]
    Clean -- yes --> Post["post_cycle hooks<br/>progress · referee · learnings"]

    Post --> Result["result =<br/>progress / no_progress / complete"]
    Fail --> Persist["write cycle_summary.json<br/>update loop_state + status.md"]
    Result --> Persist
    Persist --> End(["return summary"])

    classDef startNode fill:#e8eaed,stroke:#5f6368,color:#202124;
    classDef stepNode fill:#e8f0fe,stroke:#4285f4,color:#202124;
    classDef decisionNode fill:#fef7e0,stroke:#f9ab00,color:#202124;
    classDef failNode fill:#fce8e6,stroke:#ea4335,color:#202124;
    classDef doneNode fill:#e6f4ea,stroke:#34a853,color:#202124;
    class Start,End startNode;
    class Snap,Prompt,Restore,Run,Extract,Validate,Post stepNode;
    class Clean decisionNode;
    class Fail failNode;
    class Result,Persist doneNode;
```

### Loop lifecycle

`run_loop` evaluates stop conditions before every cycle. The common case loops
back through `RunCycle`; completion, budget limits, a stall streak, a failure
streak, or Ctrl+C end the run.

```mermaid
stateDiagram-v2
    [*] --> Idle: start / resume
    Idle --> Running: run_loop
    Running --> CheckStop: before each cycle
    CheckStop --> RunCycle: budget left
    RunCycle --> Running: apply result · update streaks
    CheckStop --> Completed: EXPERIMENT_COMPLETE
    CheckStop --> Stopped: limit · stall · failure cap
    Running --> Stopped: Ctrl+C · rollback
    Completed --> [*]
    Stopped --> [*]

    classDef run fill:#e6f4ea,stroke:#34a853,color:#202124;
    classDef halt fill:#fce8e6,stroke:#ea4335,color:#202124;
    class Idle,Running,CheckStop,RunCycle run;
    class Completed,Stopped halt;
```

## Layout

```text
loop/                  Supervisor, prompts, state, status, runner invocation
lib/                   Shared framework helpers and demo experiment packages
runners/               Runner CLIs for bundled demos
experiments/           Experiment specs, journals, results, outputs, work
.agents/skills/        Portable skills for creating experiments and notebooks
viz/                   Optional replay generation and renderer
tests/                 Framework and demo tests
```

Framework modules must not import concrete experiment packages. Experiment
behavior lives under `lib/<experiment_id>/` and is reached through runner
registries — this keeps the supervisor generic and experiments swappable.

## The cycle contract

An attempt is accepted only when every condition holds. Any failure rolls the
experiment back to its pre-cycle snapshot and retries (up to three attempts);
once attempts are exhausted the cycle is recorded as `failed`.

- Runner exits with return code `0`.
- Output contains exactly one `<promise>…</promise>` marker, and it is
  `CYCLE_DONE` or `EXPERIMENT_COMPLETE`.
- `validate_experiment` reports no actionable errors (warnings are allowed).
- `research_journal.md` changed during the cycle.
- `experiment.md` did **not** change — the spec is immutable inside a cycle.

`EXPERIMENT_COMPLETE` is validated with stricter rules and must clear the
minimum journal-cycle count declared in `experiment.md`.

## Cycle flow

1. Snapshot baselines (`research_journal.md` hash and `experiment.md`).
2. Build the prompt from static program guidance plus dynamic experiment state.
3. Invoke the configured runner with the prompt on stdin.
4. Extract assistant text from stream JSON when present, else raw stdout.
5. Check the cycle contract; retry on failure, roll back when exhausted.
6. On success, run post-cycle hooks (progress, advisory referee, learnings),
   then persist `cycle_summary.json`, `loop_state.json`, and `status.md`.

## Stopping

`run_loop` stops before a cycle when any of these hold:

- The last cycle emitted `EXPERIMENT_COMPLETE` (unless `--run-until-limit`).
- `--max-cycles` or `--max-hours` is reached.
- Three consecutive no-progress cycles (stall) — status `stalled`.
- Three consecutive failed cycles — status `failed`.

Ctrl+C stops immediately, rolls back the active cycle, and sets status
`stopped`. A per-experiment `.loop.lock` prevents two supervisors from running
the same experiment; a stale lock from a crashed run is reclaimed automatically.

## Runner configuration

Built-in presets exist for Claude, Codex, and Cursor. A custom command can be
supplied with `--runner-command`; it receives the cycle prompt on stdin and may
return stream JSON or plain text. The resolved command, model, and timeout are
persisted in loop state and attempt metadata for reproducibility.

The presets are
`claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model claude-opus-4-8-high`,
`codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high`,
and `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5`.
Override the model with `--runner-model`. Effort maps to `--effort` for Claude
and `-c model_reasoning_effort=<effort>` for Codex. Cursor has no separate
effort flag; pick a Cursor model id that already encodes effort.

## Experiment structure

```text
experiments/<experiment_id>/
  experiment.md          Immutable spec (split policy, metric, completion rules)
  research_journal.md    Per-cycle hypotheses and findings
  research_sources.md    External references gathered during research
  results.json           Candidate results
  outputs/               Deliverables a stakeholder reads
  work/                  Intermediate artifacts passed between cycles
  scripts/               One-shot scripts written during cycles
```

Use `lib.paths.outputs_dir(exp_dir)`, `work_dir(exp_dir)`, and
`scripts_dir(exp_dir)` rather than hard-coded paths; each creates the directory
on first use. The loop also writes `loop_state.json`, `status.md`, and a
`cycles/<id>/` transcript per cycle.

## Demos

The public repo ships deterministic synthetic demos that prove the harness
before pointing it at real data:

- `demo_bootstrap` — tiny classification smoke test
- `demo_classification` — synthetic binary classification
- `demo_regression` — synthetic zero-inflated revenue regression
- `demo_deep` — nonlinear tabular classification with PyTorch MLPs
  (requires `--extra deep`)

Synthetic data is generated in memory when local CSVs are absent. Generated data
files stay local and are ignored by git.
