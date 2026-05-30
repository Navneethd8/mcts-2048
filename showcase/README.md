# MCTS 2048 Showcase

This folder contains the static GitHub Pages showcase for the MCTS 2048 Mesocosm environment. It loads `data/replay.json` and renders the board state, MCTS policy table, agent action, reward, and streak metadata for each replay step.

## Preview Locally

```bash
python3 -m http.server 8080
```

Then open:

```text
http://localhost:8080/showcase/
```

## Refresh Replay Data

```bash
mesocosm run export RUN_ID -o showcase/data/replay.json
```

The bundled replay is a curated deterministic run generated from `env.py` with several move-quality patterns: clean MCTS agreement, legal lower-ranked choices, invalid action penalties, and recovery streaks.
