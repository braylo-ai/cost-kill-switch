"""Run every simulated agent with and without the kill switch; write results/simulation.json.

SIMULATED agents and illustrative prices. The guard logic is the real code under test;
spend figures are exact outputs of the token formulas in killswitch/simulator.py.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from killswitch.guard import Budget, KillSwitchTripped, RunGuard  # noqa: E402
from killswitch.simulator import AGENTS, NullGuard  # noqa: E402

BUDGET = Budget(max_usd=1.00, max_steps=50, max_depth=3)


def run(agent, guard):
    try:
        ret = agent(guard)
        reason = "hit_200_step_ceiling" if ret == "ceiling" else "completed"
    except KillSwitchTripped as e:
        reason = e.reason
    return {"outcome": reason, "steps": guard.steps, "spent_usd": round(guard.spent_usd, 4),
            "tokens": guard.tokens, "max_depth_seen": guard.max_depth_seen,
            "over_budget_usd": round(max(0.0, guard.spent_usd - BUDGET.max_usd), 4)}


def main():
    out = {"kind": "SIMULATED agents, illustrative prices (USD/1M tokens: large 15/75, medium 3/15, small 0.25/1.25)",
           "date": time.strftime("%Y-%m-%d"), "budget": BUDGET.__dict__, "unguarded_hard_ceiling_steps": 200,
           "agents": {}}
    for name, agent in AGENTS.items():
        out["agents"][name] = {"without_kill_switch": run(agent, NullGuard(budget=BUDGET)),
                               "with_kill_switch": run(agent, RunGuard(budget=BUDGET))}
        print(name, json.dumps(out["agents"][name]))
    (ROOT / "results" / "simulation.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
