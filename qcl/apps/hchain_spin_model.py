"""Stretched hydrogen chain -> antiferromagnetic Heisenberg chain: the chemistry behind xxz_dynamics.py.

A stretched, equally spaced H_N chain is a Mott insulator with antiferromagnetic
correlations (Motta et al., Phys. Rev. X 10, 031058 (2020)). Its low-energy states are
covalent: one electron per atom, coupled by superexchange, so the electronic
Hamiltonian reduces to H_spin = J(R) sum_i S_i . S_i+1 (2N spin orbitals -> N spins).
The Delta = 1 circuit of xxz_dynamics.py evolves exactly this model:
exp(-i t sum (XX + YY + ZZ)) = exp(-i 4t sum S.S), i.e. physical time t_phys = 4 t / J.

This program checks the mapping ab initio (STO-6G, as in hydrogen_chain.py):

1. J(R) = E_triplet - E_singlet of H2 by FCI;
2. spin gaps E(S) - E(0), S = 1, 2, of H_N by FCI against the Heisenberg chain with J(R);
3. spin dynamics after a quench from the covalent Neel configuration (alpha on even
   atoms, beta on odd atoms, Lowdin-orthogonalized atomic orbitals): FCI time
   evolution of the electrons against Heisenberg dynamics of the spins, and the
   Trotter circuit that ran on MQ02.

    python qcl/apps/hchain_spin_model.py --json hchain_spin.json
"""

import argparse
import json
import time

import numpy as np
from pyscf import ao2mo, fci, gto, lo, scf
from pyscf.fci import cistring
from scipy.sparse import csr_matrix, kron
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import eigsh, expm_multiply

BOHR = 0.529177210903       # angstrom
AU_TIME_FS = 0.02418884326  # femtoseconds per atomic unit of time
CIRCUIT_STEP = 0.25          # xxz_dynamics.STEP


def lattice_integrals(n_atoms, spacing, basis="sto-6g"):
    """One- and two-electron integrals in Lowdin-orthogonalized atomic orbitals."""
    atoms = "; ".join(f"H 0 0 {index * spacing * BOHR:.10f}" for index in range(n_atoms))
    mol = gto.M(atom=atoms, basis=basis, spin=n_atoms % 2, verbose=0)
    if mol.nao != n_atoms:
        raise ValueError("the spin mapping needs one orbital per atom (minimal basis)")
    orbitals = lo.orth_ao(mol, "lowdin")
    h1 = orbitals.T @ scf.hf.get_hcore(mol) @ orbitals
    eri = ao2mo.restore(1, ao2mo.full(mol, orbitals), n_atoms)
    return h1, eri, mol.energy_nuc()


def lowest(h1, eri, norb, nelec, roots=1):
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-11
    solver.max_cycle = 400
    energy, _ = solver.kernel(h1, eri, norb, nelec, nroots=roots)
    return float(np.min(energy))


def exchange(spacing, basis="sto-6g"):
    h1, eri, _ = lattice_integrals(2, spacing, basis)
    return lowest(h1, eri, 2, (2, 0)) - lowest(h1, eri, 2, (1, 1))


def heisenberg(n_spins, coupling=1.0):
    """Sparse J sum S_i . S_i+1 on an open chain (qubit i = bit i, |0> = spin up)."""
    sx = csr_matrix([[0, 0.5], [0.5, 0]])
    sy = csr_matrix([[0, -0.5j], [0.5j, 0]])
    sz = csr_matrix([[0.5, 0], [0, -0.5]])

    def site(op, index):
        out = sparse_identity(1, format="csr")
        for qubit in reversed(range(n_spins)):
            out = kron(out, op if qubit == index else sparse_identity(2), format="csr")
        return out

    total = csr_matrix((2 ** n_spins, 2 ** n_spins), dtype=complex)
    for bond in range(n_spins - 1):
        for op in (sx, sy, sz):
            total = total + site(op, bond) @ site(op, bond + 1)
    return coupling * total.real if np.allclose(total.imag.data, 0) else coupling * total


def sector_ground(hamiltonian, n_spins, n_down):
    weight = np.array([index.bit_count() for index in range(2 ** n_spins)])
    keep = np.flatnonzero(weight == n_down)
    block = hamiltonian[keep][:, keep]
    if block.shape[0] <= 2:
        return float(np.min(np.linalg.eigvalsh(block.toarray())))
    return float(eigsh(block, k=1, which="SA")[0][0])


def spin_gaps(n_atoms, spacing, coupling, basis="sto-6g"):
    h1, eri, _ = lattice_integrals(n_atoms, spacing, basis)
    half = n_atoms // 2
    electronic = [lowest(h1, eri, n_atoms, (half + spin, half - spin)) for spin in range(3)]
    model = heisenberg(n_atoms)
    spins = [sector_ground(model, n_atoms, half - spin) for spin in range(3)]
    return {"fci_gaps": [electronic[s] - electronic[0] for s in (1, 2)],
            "heisenberg_gaps": [coupling * (spins[s] - spins[0]) for s in (1, 2)]}


def neel_vector(n_atoms):
    """FCI vector of the covalent Neel determinant: alpha on even atoms, beta on odd."""
    half = n_atoms // 2
    alpha = sum(1 << site for site in range(0, n_atoms, 2))
    beta = sum(1 << site for site in range(1, n_atoms, 2))
    count = cistring.num_strings(n_atoms, half)
    vector = np.zeros((count, count))
    vector[cistring.str2addr(n_atoms, half, alpha), cistring.str2addr(n_atoms, half, beta)] = 1
    return vector


def fci_dynamics(n_atoms, spacing, times, basis="sto-6g"):
    """<S^z_i(t)> of the electrons from the Neel configuration, exact (dense FCI, S_z = 0).

    Two initial states: the bare Neel determinant, and the dressed Neel state, i.e. its
    projection on the covalent manifold (the C(N, N/2) lowest S_z = 0 states, the states
    the spin model describes), which is what a staggered pinning field prepares.
    """
    from math import comb

    h1, eri, _ = lattice_integrals(n_atoms, spacing, basis)
    half = n_atoms // 2
    nelec = (half, half)
    count = cistring.num_strings(n_atoms, half)
    addresses, hamiltonian = fci.direct_spin1.pspace(h1, eri, n_atoms, nelec, np=count * count)
    addresses = np.asarray(addresses)
    if not np.array_equal(np.sort(addresses), np.arange(count * count)):
        raise RuntimeError("pspace did not return the full space")
    strings = cistring.make_strings(range(n_atoms), half)
    sites = np.arange(n_atoms)
    alpha = (strings[addresses // count, None] >> sites) & 1
    beta = (strings[addresses % count, None] >> sites) & 1
    spin_z = 0.5 * (alpha - beta)
    energies, vectors = np.linalg.eigh(hamiltonian)
    manifold = comb(n_atoms, half)
    neel = np.zeros(count * count)
    neel_address = (cistring.str2addr(n_atoms, half, sum(1 << i for i in range(0, n_atoms, 2)))
                    * count + cistring.str2addr(n_atoms, half,
                                                sum(1 << i for i in range(1, n_atoms, 2))))
    neel[np.flatnonzero(addresses == neel_address)[0]] = 1
    amplitudes = vectors.T @ neel
    dressed = amplitudes.copy()
    dressed[manifold:] = 0
    weight = float(dressed @ dressed)
    dressed /= np.sqrt(weight)
    out = {"covalent_weight": weight,
           "spin_band_mEh": float(energies[manifold - 1] - energies[0]) * 1e3,
           "charge_gap_mEh": float(energies[manifold] - energies[manifold - 1]) * 1e3}
    for label, start in (("bare", amplitudes), ("dressed", dressed)):
        out[label] = [((np.abs(vectors @ (np.exp(-1j * energies * t) * start)) ** 2) @ spin_z
                       ).tolist() for t in times]
    return out


def heisenberg_dynamics(n_atoms, coupling, times):
    """<S^z_i(t)> of J sum S.S from the Neel state (spin up on even sites)."""
    model = heisenberg(n_atoms, coupling).tocsr()
    state = np.zeros(2 ** n_atoms, dtype=complex)
    state[sum(1 << site for site in range(1, n_atoms, 2))] = 1      # odd sites down (bit 1)
    bits = ((np.arange(2 ** n_atoms)[:, None] >> np.arange(n_atoms)) & 1)
    spin_z = 0.5 - bits
    states = expm_multiply(-1j * model, state, start=0, stop=times[-1], num=len(times),
                           endpoint=True) if len(times) > 1 else [state]
    grid = np.linspace(0, times[-1], len(times))
    if not np.allclose(grid, times):
        raise ValueError("times must be an even grid starting at 0")
    return [(np.abs(item) ** 2 @ spin_z).tolist() for item in states]


def staggered(spin_z):
    spin_z = np.asarray(spin_z)
    return float(2 * np.mean(spin_z * (-1) ** np.arange(len(spin_z))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spacings", type=float, nargs="+", default=(2.4, 2.8, 3.2, 3.6, 4.0))
    parser.add_argument("--gap-sizes", type=int, nargs="+", default=(6, 8, 10, 12))
    parser.add_argument("--dynamics-sizes", type=int, nargs="+", default=(8, 10))
    parser.add_argument("--dynamics-spacings", type=float, nargs="+", default=(3.2, 3.6))
    parser.add_argument("--circuit-time", type=float, default=1.0,
                        help="largest circuit time (steps x 0.25) to cover")
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    report = {"basis": "sto-6g", "exchange": {}, "gaps": {}, "dynamics": {}}
    for spacing in args.spacings:
        coupling = exchange(spacing)
        report["exchange"][str(spacing)] = coupling
        print(f"R = {spacing:.1f} bohr ({spacing * BOHR:.2f} A): J = {coupling * 1e3:8.3f} mEh, "
              f"circuit step dt=0.25 -> {4 * CIRCUIT_STEP / coupling * AU_TIME_FS:6.2f} fs",
              flush=True)
        for n_atoms in args.gap_sizes:
            started = time.perf_counter()
            gaps = spin_gaps(n_atoms, spacing, coupling)
            report["gaps"].setdefault(str(spacing), {})[str(n_atoms)] = gaps
            print(f"   H{n_atoms:<2d} spin gaps (mEh): FCI " +
                  " ".join(f"{g * 1e3:8.3f}" for g in gaps["fci_gaps"]) + " | Heisenberg " +
                  " ".join(f"{g * 1e3:8.3f}" for g in gaps["heisenberg_gaps"]) +
                  f"  ({time.perf_counter() - started:.0f} s)", flush=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    for spacing in args.dynamics_spacings:
        coupling = report["exchange"].get(str(spacing)) or exchange(spacing)
        circuit_times = np.linspace(0, args.circuit_time, args.points)
        physical = 4 * circuit_times / coupling
        for n_atoms in args.dynamics_sizes:
            started = time.perf_counter()
            electrons = fci_dynamics(n_atoms, spacing, physical)
            spins = heisenberg_dynamics(n_atoms, coupling, physical)
            report["dynamics"].setdefault(str(spacing), {})[str(n_atoms)] = {
                "circuit_time": circuit_times.tolist(), "time_au": physical.tolist(),
                **electrons, "heisenberg_spin_z": spins}
            errors = {label: max(abs(staggered(a) - staggered(b))
                                 for a, b in zip(electrons[label], spins))
                      for label in ("bare", "dressed")}
            moment = staggered(electrons["dressed"][0])          # moment renormalization
            errors["dressed_normalized"] = max(
                abs(staggered(a) / moment - staggered(b))
                for a, b in zip(electrons["dressed"], spins))
            print(f"R = {spacing} H{n_atoms}: covalent weight of Neel "
                  f"{electrons['covalent_weight']:.3f}, spin band "
                  f"{electrons['spin_band_mEh']:.0f} mEh, gap to ionic "
                  f"{electrons['charge_gap_mEh']:.0f} mEh; max |m_s FCI - Heisenberg| bare "
                  f"{errors['bare']:.3f}, dressed {errors['dressed']:.3f}, dressed / m_s(0) "
                  f"{errors['dressed_normalized']:.3f} over "
                  f"{physical[-1] * AU_TIME_FS:.1f} fs ({time.perf_counter() - started:.0f} s)",
                  flush=True)
            for index in range(0, len(circuit_times), len(circuit_times) // 4):
                print(f"     t = {physical[index] * AU_TIME_FS:6.2f} fs (circuit "
                      f"{circuit_times[index]:.2f}): FCI bare "
                      f"{staggered(electrons['bare'][index]):+.3f} dressed "
                      f"{staggered(electrons['dressed'][index]):+.3f}  Heisenberg "
                      f"{staggered(spins[index]):+.3f}", flush=True)
            with open(args.json, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
