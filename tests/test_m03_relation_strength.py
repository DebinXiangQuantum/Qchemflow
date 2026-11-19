import pytest

from chemflow.relations import TargetRelation, satisfies


@pytest.mark.parametrize("actual", list(TargetRelation))
@pytest.mark.parametrize("required", list(TargetRelation))
def test_relation_strength_table(actual: TargetRelation, required: TargetRelation) -> None:
    expected = (
        actual == TargetRelation.equivalent or actual == required or
        (required == TargetRelation.approximate and actual != TargetRelation.model_change)
    )
    assert satisfies(actual, required) is expected
