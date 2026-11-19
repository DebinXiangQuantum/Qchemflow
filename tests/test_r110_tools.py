import pytest

from chemflow.agent.tools import ToolRegistry, _cost_for
from chemflow.cards import load_cards
from chemflow.diagnostics import AdapterError
from chemflow.ir.schema import Node
from chemflow.ops.molecular import molecular_registry


def test_missing_card_cost_is_unknown_not_an_exception():
    cards = {card.id: card for card in load_cards()}
    for card in cards.values():
        node = Node(id="candidate", op="ground_energy", refinement=card.template)
        assert "wall_s" in _cost_for(node, molecular_registry(), cards)


@pytest.mark.parametrize("error", [TypeError("bad type"), RuntimeError("broken")])
def test_unexpected_tool_errors_are_structured(error):
    tools = ToolRegistry()

    def broken(inputs):
        raise error

    tools.register("broken", {"designer"}, broken)
    with pytest.raises(AdapterError) as caught:
        tools.call("designer", "broken")
    assert caught.value.diagnostics[0].code == "CF-E080"
    assert caught.value.__cause__ is error
