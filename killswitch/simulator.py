"""Deterministic agent simulators (SIMULATED agents, no model calls).

Each agent calls `llm(model, tokens_in, max_out)` and `guard.on_tool_call(...)` the way a
real tool-using loop would. Token counts are fixed formulas, so the guard's decisions are
exactly reproducible.
"""
from __future__ import annotations

from contextlib import contextmanager

from .guard import RunGuard, guarded

HARD_CEILING = 200  # unguarded runs stop here so the simulation terminates


class NullGuard(RunGuard):
    """Records spend but never trips: the 'no kill switch' baseline."""

    def before_llm(self, *a, **k):
        pass

    def on_tool_call(self, *a, **k):
        pass

    @contextmanager
    def subagent(self):
        self.depth += 1
        self.max_depth_seen = max(self.max_depth_seen, self.depth)
        try:
            yield self
        finally:
            self.depth -= 1

    def after_llm(self, model, tokens_in, tokens_out):
        from .guard import call_cost
        c = call_cost(model, tokens_in, tokens_out, self.prices)
        self.spent_usd += c
        self.tokens += tokens_in + tokens_out
        self.steps += 1
        return c


def _llm(model, tokens_in, max_out):
    return min(max_out, 400)


def healthy(guard: RunGuard):
    llm = guarded(guard, _llm)
    ctx = 2_000
    for i in range(6):
        llm("medium", ctx, 800)
        guard.on_tool_call("search", {"q": f"step {i}"})
        ctx += 600
    return "done"


def runaway_repeat(guard: RunGuard):
    """Retries the same failing tool call forever."""
    llm = guarded(guard, _llm)
    for _ in range(HARD_CEILING):
        llm("medium", 3_000, 800)
        guard.on_tool_call("fetch_url", {"url": "https://example.invalid/timeout"})
    return "ceiling"


def oscillate(guard: RunGuard):
    """Alternates between two tools without progress (A,B,A,B...)."""
    llm = guarded(guard, _llm)
    for i in range(HARD_CEILING):
        llm("small", 2_500, 600)
        guard.on_tool_call("open_file" if i % 2 == 0 else "close_file", {"path": "notes.txt"})
    return "ceiling"


def context_bloat(guard: RunGuard):
    """Appends every tool result to context: input tokens grow each step, cost grows quadratically."""
    llm = guarded(guard, _llm)
    ctx = 4_000
    for i in range(HARD_CEILING):
        llm("large", ctx, 1_000)
        guard.on_tool_call("read_page", {"page": i})
        ctx += 6_000
    return "ceiling"


def recursive_spawn(guard: RunGuard, level: int = 0):
    """Each sub-agent spawns two more sub-agents."""
    llm = guarded(guard, _llm)
    with guard.subagent():
        llm("medium", 2_000, 500)
        guard.on_tool_call("plan", {"level": level, "steps": guard.steps})
        if level < 6 and guard.steps < HARD_CEILING:
            recursive_spawn(guard, level + 1)
            recursive_spawn(guard, level + 1)
    return "done"


AGENTS = {
    "healthy": healthy,
    "runaway_repeat": runaway_repeat,
    "oscillate": oscillate,
    "context_bloat": context_bloat,
    "recursive_spawn": recursive_spawn,
}
