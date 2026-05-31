# Cross-Experiment Learnings

Generalizable ML insights extracted from completed experiments.

## Pattern: Start Simple

Tags: baseline, evaluation

- Compare at least one simple baseline with any more complex model before
  interpreting incremental gains.

## Pattern: Treat Close Scores Carefully

Tags: uncertainty, leaderboard

- When candidates are close, use paired bootstrap intervals or another
  uncertainty check before declaring a winner.
