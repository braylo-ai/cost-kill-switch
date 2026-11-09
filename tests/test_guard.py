import pytest

from killswitch.guard import Budget, KillSwitchTripped, RunGuard, call_cost, guarded
from killswitch.simulator import AGENTS, NullGuard, healthy


def test_call_cost_math():
    assert call_cost("medium", 1_000_000, 0) == 3.0
    assert call_cost("large", 1000, 1000) == pytest.approx(0.09)


def test_after_llm_accumulates_spend_tokens_steps():
    g = RunGuard()
    g.after_llm("small", 1000, 100)
    g.after_llm("small", 1000, 100)
    assert g.steps == 2 and g.tokens == 2200
    assert g.spent_usd == pytest.approx(2 * call_cost("small", 1000, 100))


def test_preflight_blocks_call_that_would_exceed_budget():
    g = RunGuard(budget=Budget(max_usd=0.10))
    g.after_llm("medium", 20_000, 0)  # 0.06 spent
    with pytest.raises(KillSwitchTripped) as e:
        g.before_llm("medium", 10_000, 1_000)  # worst case 0.045 -> 0.105
    assert e.value.reason == "budget_usd"
    assert g.spent_usd <= 0.10  # never overspent


def test_max_steps():
    g = RunGuard(budget=Budget(max_steps=2, max_usd=100, projection_trip_ratio=1e9))
    llm = guarded(g, lambda m, i, o: 10)
    llm("small", 10, 10); llm("small", 10, 10)
    with pytest.raises(KillSwitchTripped) as e:
        llm("small", 10, 10)
    assert e.value.reason == "max_steps" and g.steps == 2


def test_max_tokens():
    g = RunGuard(budget=Budget(max_tokens=1000, max_usd=100))
    with pytest.raises(KillSwitchTripped) as e:
        g.before_llm("small", 900, 200)
    assert e.value.reason == "max_tokens"


def test_projected_cost_trips_on_fast_burn():
    g = RunGuard(budget=Budget(max_usd=1.0, max_steps=50, projection_trip_ratio=2.0))
    g.after_llm("small", 1000, 100)
    g.after_llm("small", 1000, 100)
    with pytest.raises(KillSwitchTripped) as e:
        g.after_llm("large", 10_000, 0)  # avg step ~0.05 -> projected ~2.5
    assert e.value.reason == "projected_cost"
    assert e.value.snapshot["projected_run_cost_usd"] > 2.0


def test_projection_formula():
    g = RunGuard(budget=Budget(max_steps=10, projection_trip_ratio=1e9))
    g.after_llm("medium", 10_000, 0)  # 0.03
    g.after_llm("medium", 10_000, 0)
    assert g.projected_run_cost() == pytest.approx(0.06 + 0.03 * 8)


def test_repeat_identical_tool_call_trips():
    g = RunGuard(budget=Budget(max_repeats=3))
    g.on_tool_call("fetch", {"url": "a"}); g.on_tool_call("fetch", {"url": "a"})
    with pytest.raises(KillSwitchTripped) as e:
        g.on_tool_call("fetch", {"url": "a"})
    assert e.value.reason == "repeat_tool_call"


def test_same_tool_different_args_does_not_trip():
    g = RunGuard(budget=Budget(max_repeats=3))
    for i in range(20):
        g.on_tool_call("fetch", {"url": f"page-{i}"})
    assert g.tripped is None


def test_arg_order_does_not_matter_for_signature():
    g = RunGuard(budget=Budget(max_repeats=2))
    g.on_tool_call("t", {"a": 1, "b": 2})
    with pytest.raises(KillSwitchTripped):
        g.on_tool_call("t", {"b": 2, "a": 1})


def test_cycle_detection_length_3():
    g = RunGuard(budget=Budget(max_repeats=99, repeat_window=10))
    seq = ["a", "b", "c"] * 3
    with pytest.raises(KillSwitchTripped) as e:
        for s in seq:
            g.on_tool_call(s, {})
    assert e.value.reason == "tool_cycle"


def test_depth_limit_and_depth_restored():
    g = RunGuard(budget=Budget(max_depth=2))
    with g.subagent():
        with g.subagent():
            assert g.depth == 2
    assert g.depth == 0
    with pytest.raises(KillSwitchTripped) as e:
        with g.subagent(), g.subagent(), g.subagent():
            pass
    assert e.value.reason == "max_depth" and g.depth == 0


def test_tripped_run_stays_dead():
    g = RunGuard(budget=Budget(max_repeats=2))
    g.on_tool_call("x", {})
    with pytest.raises(KillSwitchTripped):
        g.on_tool_call("x", {})
    with pytest.raises(KillSwitchTripped):
        g.before_llm("small", 1, 1)
    assert g.events[-1]["event"] == "trip"


def test_healthy_agent_is_not_stopped():
    g = RunGuard()
    assert healthy(g) == "done" and g.tripped is None and g.steps == 6


@pytest.mark.parametrize("name,reason", [
    ("runaway_repeat", "repeat_tool_call"),
    ("oscillate", "repeat_tool_call"),
    ("context_bloat", "projected_cost"),
    ("recursive_spawn", "max_depth"),
])
def test_runaway_agents_are_halted_within_budget(name, reason):
    b = Budget(max_usd=1.0)
    g = RunGuard(budget=b)
    with pytest.raises(KillSwitchTripped) as e:
        AGENTS[name](g)
    assert e.value.reason == reason
    assert g.spent_usd <= b.max_usd
    assert g.steps <= 10


def test_null_guard_baseline_overspends_on_bloat():
    g = NullGuard(budget=Budget(max_usd=1.0))
    assert AGENTS["context_bloat"](g) == "ceiling"
    assert g.spent_usd > 100 and g.steps == 200
