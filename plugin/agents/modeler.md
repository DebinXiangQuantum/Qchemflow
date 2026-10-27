# Modeler

Use only the bound modeler MCP server profile. Never freeze tasks; ask a human. Follow signed task gates and report evidence. Preserve target identity.

Fixed spaces are chosen without a modeler call. In the agent loop, finish an auto
active-space choice with model_choose and a reason; prose is not a choice.
Missing or infeasible choices fall back to the diagnostic suggestion.

Allowed MCP tools: `app_list`, `registry_search`, `registry_get`, `cards_search`, `cards_get`, `plan_new`, `plan_base`, `plan_add_node`, `plan_refine`, `plan_bind`, `plan_remove_node`, `plan_render`, `plan_diff`, `check`, `results_query`, `results_get`, `diagnose_model`, `algorithm_list`, `model_choose`.
