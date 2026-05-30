"""2048 move-selection benchmark with a lightweight MCTS teacher."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from bench_common.env_sdk.base import BaseEnv, StepResult

MOVES = ("up", "down", "left", "right")
DEFAULT_SEED = 2048
DEFAULT_MAX_STEPS = 12
DEFAULT_ROLLOUTS = 36
DEFAULT_ROLLOUT_DEPTH = 18


def _empty_board() -> np.ndarray:
    return np.zeros((4, 4), dtype=np.int64)


def _board_to_list(board: np.ndarray) -> list[list[int]]:
    return board.astype(int).tolist()


def _max_tile(board: np.ndarray) -> int:
    return int(board.max(initial=0))


def _count_empty(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))


def _merge_line(line: np.ndarray) -> tuple[np.ndarray, int]:
    tiles = [int(value) for value in line if value]
    merged: list[int] = []
    gained = 0
    index = 0
    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            value = tiles[index] * 2
            merged.append(value)
            gained += value
            index += 2
        else:
            merged.append(tiles[index])
            index += 1
    merged.extend([0] * (4 - len(merged)))
    return np.array(merged, dtype=np.int64), gained


def _move_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    rows = []
    total_gained = 0
    for row in board:
        merged, gained = _merge_line(row)
        rows.append(merged)
        total_gained += gained
    return np.vstack(rows), total_gained


def move_board(board: np.ndarray, move: str) -> tuple[np.ndarray, int, bool]:
    if move == "left":
        moved, gained = _move_left(board)
    elif move == "right":
        moved, gained = _move_left(np.fliplr(board))
        moved = np.fliplr(moved)
    elif move == "up":
        moved, gained = _move_left(board.T)
        moved = moved.T
    elif move == "down":
        moved, gained = _move_left(np.flipud(board).T)
        moved = np.flipud(moved.T)
    else:
        return board.copy(), 0, False
    return moved, gained, bool(np.any(moved != board))


def legal_moves(board: np.ndarray) -> list[str]:
    return [move for move in MOVES if move_board(board, move)[2]]


def _spawn_tile(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = board.copy()
    empty = np.argwhere(result == 0)
    if len(empty) == 0:
        return result
    row, col = empty[int(rng.integers(0, len(empty)))]
    result[row, col] = 4 if float(rng.random()) < 0.1 else 2
    return result


def _weighted_random_move(board: np.ndarray, rng: np.random.Generator) -> str | None:
    moves = legal_moves(board)
    if not moves:
        return None
    weights = []
    for move in moves:
        next_board, gained, _ = move_board(board, move)
        weights.append(1.0 + gained / 64.0 + _count_empty(next_board) / 8.0)
    probabilities = np.array(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return str(rng.choice(moves, p=probabilities))


def _snake_monotonicity(board: np.ndarray) -> float:
    weights = np.array(
        [
            [65536, 32768, 16384, 8192],
            [512, 1024, 2048, 4096],
            [256, 128, 64, 32],
            [2, 4, 8, 16],
        ],
        dtype=float,
    )
    return float((np.log2(np.maximum(board, 1)) @ np.log2(weights).T).trace())


def _heuristic(board: np.ndarray, score: int) -> float:
    empty_bonus = _count_empty(board) * 95.0
    tile_bonus = _max_tile(board) * 1.8
    smoothness = -float(np.abs(np.diff(board, axis=0)).sum() + np.abs(np.diff(board, axis=1)).sum()) * 0.08
    return float(score + empty_bonus + tile_bonus + smoothness + _snake_monotonicity(board))


def _rollout(board: np.ndarray, rng: np.random.Generator, depth: int) -> float:
    current = board.copy()
    total_score = 0
    for _ in range(depth):
        move = _weighted_random_move(current, rng)
        if move is None:
            break
        current, gained, _ = move_board(current, move)
        total_score += gained
        current = _spawn_tile(current, rng)
    return _heuristic(current, total_score)


def mcts_policy(
    board: np.ndarray,
    rng: np.random.Generator,
    rollouts: int = DEFAULT_ROLLOUTS,
    depth: int = DEFAULT_ROLLOUT_DEPTH,
) -> pd.DataFrame:
    rows = []
    moves = legal_moves(board)
    if not moves:
        return pd.DataFrame(columns=["move", "rank", "visits", "mean_value", "immediate_score", "empty_after"])

    visits_per_move = max(1, rollouts // len(moves))
    for move in moves:
        root_board, gained, _ = move_board(board, move)
        values = []
        for _ in range(visits_per_move):
            spawned = _spawn_tile(root_board, rng)
            values.append(gained + _rollout(spawned, rng, depth))
        rows.append(
            {
                "move": move,
                "visits": visits_per_move,
                "mean_value": round(float(np.mean(values)), 3),
                "immediate_score": int(gained),
                "empty_after": _count_empty(root_board),
            }
        )

    policy = pd.DataFrame(rows).sort_values(
        ["mean_value", "immediate_score", "empty_after"],
        ascending=[False, False, False],
        ignore_index=True,
    )
    policy.insert(1, "rank", np.arange(1, len(policy) + 1))
    return policy


def _parse_action(action: Any) -> str:
    if isinstance(action, dict):
        action = action.get("move", action.get("action", ""))
    text = str(action).strip().lower()
    aliases = {
        "w": "up",
        "north": "up",
        "s": "down",
        "south": "down",
        "a": "left",
        "west": "left",
        "d": "right",
        "east": "right",
    }
    return aliases.get(text, text)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MyEnv(BaseEnv):
    def __init__(self) -> None:
        self._board = _empty_board()
        self._rng = np.random.default_rng(DEFAULT_SEED)
        self._seed = DEFAULT_SEED
        self._score = 0
        self._step_index = 0
        self._max_steps = DEFAULT_MAX_STEPS
        self._rollouts = DEFAULT_ROLLOUTS
        self._rollout_depth = DEFAULT_ROLLOUT_DEPTH
        self._correct_streak = 0
        self._wrong_streak = 0
        self._policy = pd.DataFrame()

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        self._seed = _coerce_int(seed, DEFAULT_SEED)
        self._rng = np.random.default_rng(self._seed)
        self._max_steps = max(1, _coerce_int(params.get("max_steps"), DEFAULT_MAX_STEPS))
        self._rollouts = max(4, _coerce_int(params.get("mcts_rollouts"), DEFAULT_ROLLOUTS))
        self._rollout_depth = max(2, _coerce_int(params.get("rollout_depth"), DEFAULT_ROLLOUT_DEPTH))
        self._board = _empty_board()
        self._score = 0
        self._step_index = 0
        self._correct_streak = 0
        self._wrong_streak = 0
        self._board = _spawn_tile(_spawn_tile(self._board, self._rng), self._rng)
        self._policy = mcts_policy(self._board, self._rng, self._rollouts, self._rollout_depth)
        return self._observation()

    def _observation(self) -> dict[str, Any]:
        legal = legal_moves(self._board)
        policy_records = self._policy.to_dict(orient="records")
        return {
            "game": "2048",
            "seed": self._seed,
            "step": self._step_index + 1,
            "max_steps": self._max_steps,
            "score": self._score,
            "max_tile": _max_tile(self._board),
            "empty_cells": _count_empty(self._board),
            "board": _board_to_list(self._board),
            "valid_moves": legal,
            "mcts_policy": policy_records,
            "instructions": (
                "Choose one move: up, down, left, or right. The policy table is produced by "
                "Monte Carlo tree search rollouts; the best move is rank 1. Rewards are non-binary: "
                "rank-1 moves earn increasing positive streak rewards, while lower-ranked or invalid "
                "moves receive increasing negative penalties."
            ),
        }

    def step(self, action: Any) -> StepResult:
        if self._policy.empty and not legal_moves(self._board):
            raise RuntimeError("Call reset() before step()")

        move = _parse_action(action)
        ranked_moves = {str(row.move): int(row.rank) for row in self._policy.itertuples(index=False)}
        best_move = str(self._policy.iloc[0]["move"]) if not self._policy.empty else None
        valid = move in legal_moves(self._board)

        if not valid:
            self._correct_streak = 0
            self._wrong_streak += 1
            reward = max(-2.0, -1.0 - 0.25 * self._wrong_streak)
            terminated = not legal_moves(self._board)
            info = {
                "correct": False,
                "valid": False,
                "chosen_move": move,
                "best_move": best_move,
                "rank": None,
                "reason": "invalid move",
                "correct_streak": self._correct_streak,
                "wrong_streak": self._wrong_streak,
            }
            return StepResult(
                observation=self._observation() if not terminated else {"result": "game_over"},
                reward=round(float(reward), 3),
                terminated=terminated,
                truncated=False,
                info=info,
            )

        rank = ranked_moves.get(move, len(ranked_moves) + 1)
        correct = rank == 1
        if correct:
            self._correct_streak += 1
            self._wrong_streak = 0
            reward = min(2.0, 1.0 + 0.15 * (self._correct_streak - 1))
        else:
            self._correct_streak = 0
            self._wrong_streak += 1
            reward = max(-2.0, -0.25 - 0.3 * (rank - 1) - 0.15 * self._wrong_streak)

        self._board, gained, _ = move_board(self._board, move)
        self._score += gained
        self._board = _spawn_tile(self._board, self._rng)
        self._step_index += 1
        game_over = not legal_moves(self._board)
        terminated = game_over or self._step_index >= self._max_steps or _max_tile(self._board) >= 2048
        self._policy = (
            pd.DataFrame()
            if terminated
            else mcts_policy(self._board, self._rng, self._rollouts, self._rollout_depth)
        )

        info = {
            "correct": correct,
            "valid": True,
            "chosen_move": move,
            "best_move": best_move,
            "rank": rank,
            "score_gained": int(gained),
            "total_score": int(self._score),
            "correct_streak": self._correct_streak,
            "wrong_streak": self._wrong_streak,
            "reason": "matched top MCTS move" if correct else "legal but below top MCTS move",
        }
        return StepResult(
            observation={"result": "done", "score": self._score, "max_tile": _max_tile(self._board)}
            if terminated
            else self._observation(),
            reward=round(float(reward), 3),
            terminated=terminated,
            truncated=False,
            info=info,
        )
