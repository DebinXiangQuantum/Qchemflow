"""Offline resource forecasts for every device and the remaining task budget."""

import json

from chemflow.agent.tools import PlanStore
from chemflow.backends import gpu
from chemflow.ops.molecular import molecular_registry
from chemflow.runtime.budget import RESOURCE_DIMENSIONS, BudgetLedger, resource_defaults


def show_plan(plan, registry, *, plan_id=None, ledger=None) -> str:
    lines = [f"plan_id: {plan_id}" if plan_id else "plan", f"plan_hash: {plan.hash}"]
    totals = dict.fromkeys(RESOURCE_DIMENSIONS, 0.0)
    unknown = set()
    maximum_device_time = 0.0
    for node in plan.nodes:
        binding = node.binding or node.op
        implementation = registry.implementations.get(binding)
        if implementation is None:
            implementation = gpu.catalogue().get(binding)
        if implementation is None:
            raise ValueError(f"Unknown binding: {binding}")
        declared_effect = implementation.effects
        effect = declared_effect if declared_effect != "pure" else node.effects
        predicted = implementation.resource_model(**node.params)
        kind = implementation.backend_kind
        predicted = resource_defaults(predicted, kind)
        line = f"{node.id} {node.op} binding={binding} effect={effect}"
        for dimension in RESOURCE_DIMENSIONS:
            amount = predicted.get(dimension)
            line += f" predicted_{dimension}={amount if amount is not None else 'unknown'}"
            if amount is None:
                unknown.add(dimension)
            else:
                totals[dimension] += amount
        diagnostics = predicted.get("diagnostics", {})
        if kind == "qpu":
            for name in ("qpu_s_p99", "qpu_s_max_observed"):
                line += f" {name}={diagnostics.get(name, 'unknown')}"
            maximum_device_time += diagnostics.get("qpu_s_max_observed") or 0
        lines.append(line)
    for dimension, amount in totals.items():
        suffix = " + unknown" if dimension in unknown else ""
        lines.append(f"total predicted_{dimension}={amount:g}{suffix}")
    if ledger is None:
        ledger = BudgetLedger(plan.budget)
    ledger.reload()
    for dimension, limit in ledger.limits.items():
        spent = ledger.spent[dimension]
        spent_text = f"{spent:g}" if spent is not None else "unknown"
        remaining = max(0, limit - spent) if spent is not None else 0
        lines.append(f"budget {dimension} limit={limit:g} spent={spent_text} "
                     f"remaining={remaining:g}")
        if dimension == "qpu_s" and remaining < maximum_device_time:
            lines.append(f"WARNING: remaining qpu_s={remaining:g} is below the plan's "
                         f"maximum observed device time {maximum_device_time:g}")
    return "\n".join(lines)


def show_stored_plan(plan_id, plan_dir, qpu=None):
    plans = PlanStore(plan_dir)
    plan = plans.get(plan_id)
    registry = molecular_registry()
    if any(node.binding == "aer_noisy_mq02" for node in plan.nodes):
        from chemflow.backends.aer_noise import register_noisy_aer
        from chemflow.backends.calibration import load_calibration

        register_noisy_aer(registry, load_calibration())
    if qpu is not None or any(node.binding == "lqcloud_mq02" for node in plan.nodes):
        from chemflow.backends.lqcloud import LqcloudBackend, register_lqcloud

        backend = qpu or "MQ02"
        register_lqcloud(registry, LqcloudBackend(backend=backend),
                         identifier=f"lqcloud_{backend.lower()}")
    ledger = None
    context_path = plan_dir / f"{plan_id}.resources.json"
    if context_path.exists():
        context = json.loads(context_path.read_text())
        ledger = BudgetLedger(context["budget"],
                              plan_dir / f"{context['task_hash']}.ledger.json")
    return show_plan(plan, registry, plan_id=plan_id, ledger=ledger)
