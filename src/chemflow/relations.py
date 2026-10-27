from enum import StrEnum


class TargetRelation(StrEnum):
    equivalent = "equivalent"
    variational_upper_bound = "variational_upper_bound"
    stochastic_estimate = "stochastic_estimate"
    approximate = "approximate"
    model_change = "model_change"


# Rows are upstream relation, columns downstream. Equivalent is identity;
# model_change absorbs; noisy variational bounds lose their ordering guarantee.
RELATION_TABLE: dict[tuple[TargetRelation, TargetRelation], TargetRelation] = {}
for upstream in TargetRelation:
    for downstream in TargetRelation:
        if TargetRelation.model_change in (upstream, downstream):
            result = TargetRelation.model_change
        elif upstream == TargetRelation.equivalent:
            result = downstream
        elif downstream == TargetRelation.equivalent:
            result = upstream
        elif upstream == downstream == TargetRelation.variational_upper_bound:
            result = TargetRelation.variational_upper_bound
        elif upstream == downstream == TargetRelation.stochastic_estimate:
            result = TargetRelation.stochastic_estimate
        else:
            result = TargetRelation.approximate
        RELATION_TABLE[upstream, downstream] = result


def compose(upstream: TargetRelation, downstream: TargetRelation) -> TargetRelation:
    return RELATION_TABLE[upstream, downstream]


def satisfies(actual: TargetRelation, required: TargetRelation) -> bool:
    if actual == TargetRelation.equivalent or actual == required:
        return True
    return required == TargetRelation.approximate and actual != TargetRelation.model_change


def combine_difference(left: TargetRelation, right: TargetRelation) -> TargetRelation:
    """Subtract relations: exact/exact=exact, upper/exact=upper,
    exact/upper=approximate (lower bound unrepresented), upper/upper=approximate;
    stochastic with exact or stochastic=stochastic, otherwise approximate;
    model_change on either side=model_change.
    """
    if TargetRelation.model_change in (left, right):
        return TargetRelation.model_change
    if left == right == TargetRelation.equivalent:
        return TargetRelation.equivalent
    if (left == TargetRelation.variational_upper_bound and
        right == TargetRelation.equivalent):
        return TargetRelation.variational_upper_bound
    if (TargetRelation.stochastic_estimate in (left, right) and
        {left, right} <= {TargetRelation.equivalent, TargetRelation.stochastic_estimate}):
        return TargetRelation.stochastic_estimate
    return TargetRelation.approximate


def combine_linear(terms: list[tuple[float, TargetRelation]]) -> TargetRelation:
    relations = {relation for _, relation in terms}
    if TargetRelation.model_change in relations:
        return TargetRelation.model_change
    if relations <= {TargetRelation.equivalent}:
        return TargetRelation.equivalent
    if relations <= {TargetRelation.equivalent, TargetRelation.stochastic_estimate}:
        return TargetRelation.stochastic_estimate
    if all(relation == TargetRelation.equivalent or
           (relation == TargetRelation.variational_upper_bound and coefficient > 0)
           for coefficient, relation in terms):
        return TargetRelation.variational_upper_bound
    return TargetRelation.approximate


def ritz_relation(full_coverage: bool) -> TargetRelation:
    """HUM: Ritz root k bounds sector eigenvalue k, irrespective of how P was found."""
    return (TargetRelation.equivalent if full_coverage else
            TargetRelation.variational_upper_bound)


def response_relation(full_coverage: bool) -> TargetRelation:
    """Exact response requires complete sector states and transition amplitudes."""
    return TargetRelation.equivalent if full_coverage else TargetRelation.approximate
