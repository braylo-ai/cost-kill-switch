"""Run guard: tracks loop depth, steps, tokens and cost per agent run, and halts the run
BEFORE a call that would break a budget (pre-flight projection), not after."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

# Illustrative prices in USD per 1M tokens. Not any vendor's price list; set your own.
PRICES = {
    "large": {"in": 15.0, "out": 75.0},
    "medium": {"in": 3.0, "out": 15.0},
    "small": {"in": 0.25, "out": 1.25},
}


def call_cost(model: str, tokens_in: int, tokens_out: int, prices: dict = PRICES) -> float:
    p = prices[model]
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000


@dataclass
class Budget:
    max_usd: float = 1.00
    max_steps: int = 50
    max_depth: int = 3
    max_tokens: int = 500_000
    max_repeats: int = 3            # identical tool call (name+args) seen this many times in window
    repeat_window: int = 10
    max_cycle_len: int = 4          # detect A,B,A,B... cycles up to this length
    projection_trip_ratio: float = 2.0  # halt if projected run cost > ratio * max_usd


class KillSwitchTripped(RuntimeError):
    def __init__(self, reason: str, detail: str, snapshot: dict):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail, self.snapshot = reason, detail, snapshot


@dataclass
class RunGuard:
    budget: Budget = field(default_factory=Budget)
    prices: dict = field(default_factory=lambda: PRICES)
    spent_usd: float = 0.0
    tokens: int = 0
    steps: int = 0
    depth: int = 0
    max_depth_seen: int = 0
    tripped: KillSwitchTripped | None = None
    _recent: deque = field(default_factory=lambda: deque(maxlen=32))
    events: list = field(default_factory=list)

    # ---------- snapshot / projection ----------
    def projected_run_cost(self) -> float:
        """Spend so far + average cost per step * steps left under max_steps."""
        if self.steps == 0:
            return self.spent_usd
        avg = self.spent_usd / self.steps
        return self.spent_usd + avg * max(0, self.budget.max_steps - self.steps)

    def snapshot(self) -> dict:
        return {
            "spent_usd": round(self.spent_usd, 6),
            "tokens": self.tokens,
            "steps": self.steps,
            "depth": self.depth,
            "max_depth_seen": self.max_depth_seen,
            "projected_run_cost_usd": round(self.projected_run_cost(), 6),
        }

    def _trip(self, reason: str, detail: str):
        self.tripped = KillSwitchTripped(reason, detail, self.snapshot())
        self.events.append({"event": "trip", "reason": reason, "detail": detail, **self.snapshot()})
        raise self.tripped

    def _check_alive(self):
        if self.tripped is not None:
            raise self.tripped  # a tripped run stays dead

    # ---------- hooks ----------
    def before_llm(self, model: str, tokens_in: int, max_tokens_out: int) -> None:
        """Pre-flight: refuse the call if its worst case would exceed a hard limit."""
        self._check_alive()
        b = self.budget
        if self.steps + 1 > b.max_steps:
            self._trip("max_steps", f"step {self.steps + 1} > {b.max_steps}")
        worst = call_cost(model, tokens_in, max_tokens_out, self.prices)
        if self.spent_usd + worst > b.max_usd:
            self._trip("budget_usd", f"spent {self.spent_usd:.4f} + worst-case {worst:.4f} > {b.max_usd}")
        if self.tokens + tokens_in + max_tokens_out > b.max_tokens:
            self._trip("max_tokens", f"{self.tokens} + {tokens_in + max_tokens_out} > {b.max_tokens}")

    def after_llm(self, model: str, tokens_in: int, tokens_out: int) -> float:
        cost = call_cost(model, tokens_in, tokens_out, self.prices)
        self.spent_usd += cost
        self.tokens += tokens_in + tokens_out
        self.steps += 1
        self.events.append({"event": "llm", "model": model, "in": tokens_in, "out": tokens_out, "cost": cost})
        if self.steps >= 3 and self.projected_run_cost() > self.budget.projection_trip_ratio * self.budget.max_usd:
            self._trip("projected_cost", f"projected {self.projected_run_cost():.4f} > "
                       f"{self.budget.projection_trip_ratio}x budget {self.budget.max_usd}")
        return cost

    def on_tool_call(self, name: str, args: dict) -> None:
        self._check_alive()
        sig = hashlib.sha256(f"{name}:{json.dumps(args, sort_keys=True)}".encode()).hexdigest()[:16]
        self._recent.append(sig)
        window = list(self._recent)[-self.budget.repeat_window:]
        if window.count(sig) >= self.budget.max_repeats:
            self._trip("repeat_tool_call", f"{name} with identical args seen {window.count(sig)}x "
                       f"in last {len(window)} calls")
        seq = list(self._recent)
        for k in range(2, self.budget.max_cycle_len + 1):
            if len(seq) >= 3 * k and seq[-k:] == seq[-2 * k:-k] == seq[-3 * k:-2 * k]:
                self._trip("tool_cycle", f"cycle of length {k} repeated 3x")

    @contextmanager
    def subagent(self):
        self._check_alive()
        self.depth += 1
        self.max_depth_seen = max(self.max_depth_seen, self.depth)
        try:
            if self.depth > self.budget.max_depth:
                self._trip("max_depth", f"depth {self.depth} > {self.budget.max_depth}")
            yield self
        finally:
            self.depth -= 1


def guarded(guard: RunGuard, llm_fn):
    """Wrap `llm_fn(model, prompt_tokens, max_out) -> tokens_out` with the guard hooks."""
    def call(model: str, tokens_in: int, max_out: int) -> int:
        guard.before_llm(model, tokens_in, max_out)
        out = llm_fn(model, tokens_in, max_out)
        guard.after_llm(model, tokens_in, out)
        return out
    return call
