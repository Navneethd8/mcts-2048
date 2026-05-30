# MCTS 2048

Mesocosm / BenchAnything environment for evaluating 2048 move selection against a Monte Carlo tree search teacher.

Each episode starts from a seeded 2048 board and runs for up to 12 moves. On every turn, the observation includes:

- The current 4x4 board, score, max tile, and legal moves.
- A compact MCTS policy table with move rank, visits, estimated value, immediate score, and empty-cell count.
- Instructions asking the agent to return exactly one move: `up`, `down`, `left`, or `right`.

The environment uses only `numpy` and `pandas` beyond the Mesocosm adapter SDK. Rewards are non-binary: the rank-1 MCTS move earns a positive reward that grows with a correct streak, while lower-ranked legal moves and invalid moves receive increasingly negative penalties.

## Showcase

The static GitHub Pages showcase lives in `showcase/` and replays a curated deterministic run:

```text
https://navneethd8.github.io/mcts-2048/
```

## Local Run

```bash
pip install swecc-mesocosm
pip install -r requirements.txt
python adapter.py
mesocosm run local --episodes 3
```

You can tune local difficulty with scenario parameters:

```bash
mesocosm run local --scenario '{"max_steps": 8, "mcts_rollouts": 24, "rollout_depth": 12}'
```
