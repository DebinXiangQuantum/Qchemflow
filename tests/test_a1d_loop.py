

def test_doubles_closure_has_exact_two_layer_support():
    from chemflow.domains.algorithms import extend_subspace
    from chemflow.types import Sector, Subspace

    source = Subspace(sector=Sector(n_alpha=2, n_beta=2, n_orb=5),
                      determinants=(0b00110 | (0b00011 << 5),))
    target = Sector(n_alpha=3, n_beta=2, n_orb=5)
    image = source.determinants[0] | 1
    result = extend_subspace(None, source, target, "create", 0, "alpha", closure="doubles")
    expected = {state for state in range(1 << 10)
                if (state & 31).bit_count() == 3 and (state >> 5).bit_count() == 2
                and (state ^ image).bit_count() <= 4}
    assert set(result.determinants) == expected
