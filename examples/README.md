# Examples

Artifacts the loop produces, committed so you can look without running anything.

## `demo_bootstrap_replay.html`

A self-contained replay of the `demo_bootstrap` experiment — open it in any
browser (no server, no dependencies). It walks through the five research cycles
(hypothesis → research → training → scoreboard → journal), building up the
leaderboard cycle by cycle.

It is generated from the committed experiment artifacts:

```bash
uv run python viz/generate.py experiments/demo_bootstrap
uv run python viz/bundle.py viz/output/demo_bootstrap/script.json examples/demo_bootstrap_replay.html
```

Source of truth: [`experiments/demo_bootstrap/research_journal.md`](../experiments/demo_bootstrap/research_journal.md)
and [`results.json`](../experiments/demo_bootstrap/results.json).

## Benchmark report

A runner benchmark writes `comparison.md` / `comparison.csv` (ranked by referee
score, then leaderboard). Generate one with:

```bash
uv run python -m loop bench experiments/demo_bootstrap --runners claude,codex,cursor --max-cycles 6
```

Ad-hoc benchmark output under `bench/` is gitignored; copy a run here if you want
to keep it as a reference.
