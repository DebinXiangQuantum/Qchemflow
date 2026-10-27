from collections import defaultdict

from chemflow.runtime.artifacts import ArtifactStore
from chemflow.types import Hamiltonian

_PRODUCT = {
    ("X", "Y"): ("Z", 1j), ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j), ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j), ("X", "Z"): ("Y", -1j),
}


def multiply(left: str, right: str) -> tuple[str, complex]:
    output: list[str] = []
    phase = 1 + 0j
    for first, second in zip(left, right):
        if first == "I":
            output.append(second)
        elif second == "I":
            output.append(first)
        elif first == second:
            output.append("I")
        else:
            result, multiplier = _PRODUCT[first, second]
            output.append(result)
            phase *= multiplier
    return "".join(output), phase


def ladder(index: int, count: int, create: bool) -> dict[str, complex]:
    prefix = "Z" * index
    suffix = "I" * (count - index - 1)
    return {prefix + "X" + suffix: 0.5,
            prefix + "Y" + suffix: (-0.5j if create else 0.5j)}


def product(left: dict[str, complex], right: dict[str, complex]) -> dict[str, complex]:
    output: dict[str, complex] = defaultdict(complex)
    for first, first_value in left.items():
        for second, second_value in right.items():
            word, phase = multiply(first, second)
            output[word] += first_value * second_value * phase
    return output


def pauli_terms(hamiltonian: Hamiltonian, store: ArtifactStore) -> dict[str, float]:
    """Explicit Jordan–Wigner expansion; word position 0 is least-significant qubit."""
    n_orb = hamiltonian.n_orb
    count = 2 * n_orb
    h1 = store.get(hamiltonian.h1_ref)
    eri = store.get(hamiltonian.h2_ref)
    terms: dict[str, complex] = defaultdict(complex)
    terms["I" * count] = complex(hamiltonian.ledger.total().value)
    for annihilator in range(count):
        for creator in range(count):
            if creator // n_orb != annihilator // n_orb:
                continue
            coefficient = h1[creator % n_orb, annihilator % n_orb]
            if abs(coefficient) > 1e-14:
                for word, value in product(ladder(creator, count, True),
                                           ladder(annihilator, count, False)).items():
                    terms[word] += coefficient * value
    for annihilator in range(count):
        for second_annihilator in range(count):
            for creator in range(count):
                if creator // n_orb != annihilator // n_orb:
                    continue
                for second_creator in range(count):
                    if second_creator // n_orb != second_annihilator // n_orb:
                        continue
                    coefficient = 0.5 * eri[creator % n_orb, annihilator % n_orb,
                                            second_creator % n_orb,
                                            second_annihilator % n_orb]
                    if abs(coefficient) < 1e-14:
                        continue
                    factors = [ladder(creator, count, True),
                               ladder(second_creator, count, True),
                               ladder(second_annihilator, count, False),
                               ladder(annihilator, count, False)]
                    expanded = {"I" * count: 1 + 0j}
                    for factor in factors:
                        expanded = product(expanded, factor)
                    for word, value in expanded.items():
                        terms[word] += coefficient * value
    if any(abs(value.imag) > 1e-9 for value in terms.values()):
        raise ValueError("JW expansion is non-Hermitian")
    return {word: float(value.real) for word, value in terms.items()
            if abs(value.real) > 1e-13}
