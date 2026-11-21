import pytest

from chemflow.relations import TargetRelation as R
from chemflow.relations import combine_difference


@pytest.mark.parametrize("left,right,expected", [
    (left, right, expected)
    for left, row in zip(R, (
        (R.equivalent, R.approximate, R.stochastic_estimate, R.approximate, R.model_change),
        (R.variational_upper_bound, R.approximate, R.approximate, R.approximate,
         R.model_change),
        (R.stochastic_estimate, R.approximate, R.stochastic_estimate, R.approximate,
         R.model_change),
        (R.approximate, R.approximate, R.approximate, R.approximate, R.model_change),
        (R.model_change, R.model_change, R.model_change, R.model_change, R.model_change),
    ))
    for right, expected in zip(R, row)
])
def test_difference_relation_table(left: R, right: R, expected: R) -> None:
    assert combine_difference(left, right) == expected
