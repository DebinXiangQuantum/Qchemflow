"""Mathematical rules, separate from falsifiable implementation contracts."""


def _lemma(statement, conditions, reference):
    return {"statement": statement, "side_conditions": conditions, "reference": reference}


LEMMAS = {
    "empirical_model_evaluation": _lemma(
        "Evaluation of the declared empirical formula is equivalent to its model-defined "
        "quantity, not the experimental property (model_change).",
        ["empirical_evaluation_matches"],
        "Definitional equality; Delaney, JCICS 2004, 44, 1000; "
        "Yalkowsky and Valvani, J. Pharm. Sci. 1980, 69, 912–922"),
    "declared_model_evaluation": _lemma(
        "Evaluation of the declared solvent or orbital model equals that model-defined "
        "target, without asserting equality to a physical observable.",
        ["declared_model_matches"], "Definitional equality; ChemFlow declared-model semantics"),
    "energy_collection_exact": _lemma(
        "A finite collection of signed linear combinations of exact component energies "
        "equals the corresponding collection of target energies. Sorting exact item energies "
        "gives their energy order, without asserting separation of ties.",
        ["collection_inputs"], "Linearity over R applied componentwise; definition of sorting"),
    "exact_diagonalization": _lemma(
        "For Hermitian H on a nontrivial finite-dimensional sector S, the infimum of "
        "the nonzero-vector Rayleigh quotient is an eigenvalue and is <= every eigenvalue. "
        "Returning this value is the separate diagonalization implementation contract.",
        ["hermitian", "sector_matches", "spin_supported"], "Courant–Fischer min-max theorem"),
    "sector_decomposition": _lemma(
        "If H commutes with N, it maps each eigenspace of N into itself. Applying this "
        "to N_alpha and N_beta gives joint-sector invariance. Establishing commutation "
        "for the encoded number-conserving Hamiltonian is an implementation obligation.",
        ["number_conserving", "sector_matches"], "Szabo–Ostlund, Modern Quantum Chemistry, ch. 2"),
    "rayleigh_ritz": _lemma(
        "For a nontrivial subspace V of a finite-dimensional sector S and Hermitian H, "
        "the infimum Rayleigh quotient on V is >= that on S; it is the lowest "
        "eigenvalue of the orthogonal compression to V. This rule covers only k=0.",
        ["hermitian", "basis_in_sector", "spin_supported"],
        "Hylleraas–Undheim (1930); MacDonald (1933); Courant–Fischer"),
    "rayleigh_ritz_full": _lemma(
        "For Hermitian H on a nontrivial finite-dimensional sector S, V=S gives "
        "equality of the lowest Ritz value and the ground energy. Establishing V=S "
        "from the exported determinant basis is an implementation obligation.",
        ["hermitian", "basis_in_sector", "full_basis", "spin_supported"], "Rayleigh–Ritz theorem"),
    "variational_principle": _lemma(
        "For nonzero psi in invariant sector S, <psi,H psi>/<psi,psi> >= lambda_0(H|S). "
        "A normalized HF determinant is one such trial state; convergence is not required.",
        ["hermitian", "sector_matches", "trial_in_sector", "exact_evaluation", "spin_supported"],
        "Szabo–Ostlund, Modern Quantum Chemistry, ch. 3, variational principle"),
    "positive_linear_combination": _lemma(
        "If x_i >= t_i for inexact terms and c_i >= 0 for those terms, "
        "sum(c_i*x_i) >= sum(c_i*t_i); exact terms may have either sign.",
        ["linear_inputs"], "Order compatibility of addition and nonnegative multiplication over R"),
    "linear_combination_exact": _lemma(
        "If x_i=t_i for every nonzero coefficient, sum(c_i*x_i)=sum(c_i*t_i).",
        ["linear_inputs"], "Linearity over R"),
    "stochastic_linear_combination": _lemma(
        "A deterministic finite linear combination of exact values and sample-mean "
        "estimators is classified stochastic_estimate for the corresponding linear "
        "target; this is not a proof of unbiasedness, integrability or confidence.",
        ["linear_inputs"], "Linearity of expectation"),
    "lehmann_completeness": _lemma(
        "Given a finite orthonormal eigenbasis of Hermitian H with real eigenvalues E_n "
        "and z-E_n != 0 for all n, (z-H)u=v has a unique solution and "
        "<v,u> = sum_n |<n,v>|^2/(z-E_n). Im(z)>0 suffices. Substituting "
        "v=c†|0> and v=c|0> with the particle/hole shifted Hamiltonians gives the "
        "implemented Lehmann sum; the sector/ladder construction is trusted.",
        ["complete_response"], "Fetter–Walecka, Quantum Theory of Many-Particle Systems, Lehmann representation"),
    "stochastic_estimate": _lemma(
        "Finite sample averages of the specified observable are stochastic estimates, "
        "not deterministic variational bounds; no accuracy or confidence level follows.",
        ["sampled_evaluation"], "Sample-mean definition; linearity of expectation"),
    "expectation_evaluation": _lemma(
        "Exact evaluation of the specified observable in the specified state equals "
        "that expectation target; it does not equal the ground energy in general.",
        ["known_denotation"], "Definition of the quantum expectation value"),
    "rrho_evaluation": _lemma(
        "Evaluating the specified rigid-rotor harmonic-oscillator ideal-gas correction "
        "equals that model's correction at fixed temperature, pressure and frequencies; "
        "this does not assert equality to experimental thermodynamics.",
        ["thermal_contract"], "McQuarrie, Statistical Mechanics, ideal-gas partition functions"),
    "no_ordering_law": _lemma(
        "A denoted approximation without applicable equality or order hypotheses is "
        "classified approximate. This asserts neither an error bound nor an ordering.",
        ["known_denotation"], "ChemFlow relation calculus: conservative forgetting rule"),
    "pointwise_equality": _lemma(
        "An ordered tuple of exact component values equals the corresponding tuple target. "
        "This does not certify interpolation between coordinates or geometry optimality.",
        ["curve_inputs"], "Extensional equality of finite tuples, component by component"),
}


LEMMAS["rayleigh_ritz_excited"] = _lemma(
    "For Hermitian H and nontrivial V subset S, each ordered Ritz root k>0 with "
    "k<dim(V) is >= the corresponding sector eigenvalue; when V=S all roots agree.",
    ["hermitian", "basis_in_sector", "spin_supported", "excited_roots"],
    "Courant–Fischer; Hylleraas–Undheim (1930); MacDonald (1933)")
LEMMAS["weinstein_bound"] = _lemma(
    "For Hermitian H on a finite-dimensional complex inner product space and unit psi, "
    "with mu=Re< H psi,psi >, some eigenvalue lambda satisfies "
    "|lambda-mu| <= sqrt(||H psi||^2-mu^2). This does not identify a ground eigenvalue.",
    ["hermitian", "normalized_trial", "exact_moments"], "Weinstein residual bound")

_DEFINITIONAL = {
    "pointwise_equality": "Equality of the ordered curve to its tuple target is defined "
    "component by component, conditional on exact inputs and the curve binding contract; "
    "it does not certify interpolation, geometry optimality or physical accuracy.",
    "energy_collection_exact": "The collection is defined by signed component-energy sums "
    "and their sorted order. Equality is conditional on exact component inputs and the "
    "collection binding contract; it does not certify tie separation or physical accuracy.",
    "declared_model_evaluation": "Equality to the declared solvent or frontier-orbital model "
    "is its denotation and binding contract, not a proof of physical accuracy or of the "
    "numerical evaluator.",
    "model_evaluation": "Equality only to the declared finite model evaluation, including its "
    "sampling seed; the numerical implementation is trusted. No experimental equality is implied.",
    "stochastic_estimate": "Classifies a finite sample mean; asserts no accuracy, unbiasedness "
    "or deterministic ordering. Sampling-law correctness is an implementation obligation.",
    "stochastic_linear_combination": "Classifies a deterministic finite combination of "
    "estimates. No machine-checked probabilistic claim: expectation linearity additionally "
    "requires integrability and correct marginal sampling laws; independence is unnecessary.",
    "expectation_evaluation": "Equality to the specified state/observable expectation is "
    "the denotation and binding contract, not a theorem about the numerical evaluator.",
    "rrho_evaluation": "Equality to the fixed RRHO model is the denotation and binding "
    "contract, not a proof of physical accuracy or of the numerical partition function.",
    "no_ordering_law": "Conservative classification: absence of certified equality/order "
    "yields approximate, without asserting that an ordering is mathematically impossible.",
    "empirical_model_evaluation": "Evaluating a declared empirical formula is its "
    "model-defined quantity by definition; agreement with experiment is model_change and is "
    "characterised by the model card's dataset metrics, not proved.",
}
# Fail closed: a lemma is machine-checked only if it is listed here and in the manifest.
_MACHINE_CHECKED = {"exact_diagonalization", "sector_decomposition", "rayleigh_ritz",
                    "rayleigh_ritz_full", "variational_principle", "positive_linear_combination",
                    "linear_combination_exact", "lehmann_completeness", "weinstein_bound"}
LEMMAS["model_evaluation"] = _lemma(
    "Evaluation of a fixed, explicitly model-defined protocol equals that protocol's output, "
    "not an experimental free energy. For sampled protocols the finite seeded estimator is "
    "the target, not the ensemble limit.", ["model_contract"],
    "Definition of single-trajectory MM/GBSA; Genheden and Ryde, Expert Opin Drug Discov "
    "10 (2015), 449–461, doi:10.1517/17460441.2015.1032936")
for _name, _entry in LEMMAS.items():
    if _name in _DEFINITIONAL:
        _entry.update(lean_status="definitional", lean=None, lean_note=_DEFINITIONAL[_name])
    elif _name == "rayleigh_ritz_excited":
        _entry.update(lean_status="cited", lean=None, lean_note="The higher-root min-max "
                      "theorem and full-spectrum equality are cited, not formalized here.")
    elif _name not in _MACHINE_CHECKED:
        raise ValueError(f"Lemma {_name} has no formal status: classify it as "
                         "definitional, cited or machine-checked")
    else:
        _entry.update(lean_status="machine_checked", lean=f"ChemflowFormal.{_name}",
                      lean_note="Finite-dimensional exact mathematics; numerical binding "
                      "contracts and floating-point side-condition checks remain trusted.")


AXIOMS = {}


def _contract(binding, op, kind, statement):
    AXIOMS[f"binding:{binding}"] = {
        "name": f"binding:{binding}", "binding": binding, "op": op, "kind": kind,
        "version": "1", "statement": statement}


for _binding in ("logs_esol", "logs_gse", "empirical_estimate"):
    _contract(_binding, _binding, "empirical", "Evaluates the specified ESOL or clamped GSE "
              "formula, with the bundled dataset and RDKit descriptors; not a physical identity.")
_contract("metabolic_sites", "metabolic_sites", "declared_model", "Evaluates the model-defined maximum "
          "C-H SASA_fraction * sigmoid((90 - BDE_kcal/mol)/3), using the declared electronic "
          "method and FreeSASA; equality to this declared model only, not experimental metabolism.")
_contract("dmrg_block2", "ground_energy", "trial", "Reports the expectation of the physical "
          "Hamiltonian in the final normalized block2 MPS, in the requested number sector "
          "and SU(2) spin irrep when specified. No extrapolated energy is reported.")
_contract("correlation_function", "correlation_function", "expectation", "Evaluates explicit "
          "fermion algebra in the exported normalized site-basis sector vector. This equals "
          "the ground-state correlation target only when the source is the exact ground state; "
          "subspace and mean-field correlations have no ordering law.")
_contract("allen_dynes_tc", "allen_dynes_tc", "opaque", "Evaluates the Allen-Dynes (McMillan) "
          "model-defined Tc in kelvin from user/EPW inputs, not an ab initio prediction of Tc.")
_contract("dft_pyscf", "ground_energy", "opaque", "Full-space RKS/UKS total energy plus "
          "functional-specific D3(BJ)/D4 dispersion; no physical ordering law.")
for _binding in ("solvation_free_energy", "frontier_orbitals"):
    _contract(_binding, _binding, "declared_model", "Evaluates exactly the declared model: "
              "solvent-minus-gas HF/DFT IEF-PCM/SMD or GFN2 ALPB/GBSA energy, or converged "
              "HF/DFT spin-channel frontier orbital energy/gap; not a physical observable.")
_contract("charge_energy_difference", "charge_energy_difference", "linear",
          "Signed difference of physical total energies in the specified charge sectors.")
_contract("eom_charge_energy", "eom_charge_energy", "opaque",
          "Restricted EOM-IP/EA-CCSD charged excitation; EA has reversed attachment-energy sign.")
for _binding in ("protein_structure", "ligand_structure", "complex", "pocket", "ligand_parameters"):
    _contract(_binding, _binding, "data", "Produces the documented prepared structure, pocket or "
              "force-field parameters; preparation assumptions and charge states are recorded.")
_contract("mmgbsa", "mmgbsa", "model", "Evaluates Amber14/GBn2 single-trajectory E_complex minus "
          "E_receptor minus E_ligand at the same coordinates with the declared ligand parameters; "
          "returns the finite seeded frame mean when requested. Experimental relation: model_change.")
_contract("fragment_interaction", "fragment_interaction", "opaque", "Evaluates a pairwise-additive "
          "sum of capped side-chain ligand counterpoise energies; no binding-energy ordering law.")


for _binding in ("fci_pyscf", "sector_diag_numpy"):
    _contract(_binding, "ground_energy", "exact", "Returns the lowest physical energy in "
              "the requested particle and, if specified, total-spin sector, including constants.")
for _binding in ("sci_pyscf", "sqd_sector"):
    _contract(_binding, "ground_energy", "ritz", "Returns the lowest projected physical "
              "energy on the exported determinant set, with requested spin selection when "
              "spin completion succeeds; the artifact is the actual final solve basis.")
for _binding in ("vqe_lqcloud", "vqe_aer_mq02", "vqe_shots_ideal"):
    _contract(_binding, "ground_energy", "sample", "Returns a final finite-shot Pauli "
              "estimate of <psi(theta)|H|psi(theta)> after circuit optimization; "
              "not a variational bound or a guarantee of ground-state accuracy.")
for _binding in ("vqe_statevector", "adapt_statevector"):
    _contract(_binding, "ground_energy", "trial", "Uses the exported normalized state in the "
              "fixed-(N_alpha,N_beta) SectorEngine; exact mode evaluates its Rayleigh quotient, "
              "shot mode evaluates sampled Pauli terms, not a variational bound.")
for _binding in ("hf", "reaction_hf", "lattice_hf"):
    _contract(_binding, "ground_energy", "hf", "Evaluates the physical Hamiltonian on a "
              "normalized HF determinant with the requested alpha and beta occupations.")
for _binding in ("mp2", "ccsd", "ccsd_t", "reaction_mp2", "reaction_ccsd",
                 "reaction_ccsd_t", "ewf_embedding"):
    _contract(_binding, "ground_energy", "opaque", "Computes the named correlation or "
              "embedding approximation for the input Hamiltonian; no ordering contract.")
_contract("rayleigh_ritz", "rayleigh_ritz", "ritz", "Diagonalizes the Hermitian projected "
          "Hamiltonian on the supplied determinant basis, retaining the full projected resolution.")
_contract("lehmann_from_spectra", "spectral_function", "response", "Evaluates the retarded "
          "Lehmann sum using the supplied resolutions and fermionic ladder amplitudes.")
_contract("spectral_exact", "spectral_function", "response_exact", "Constructs full ground "
          "and feasible neighboring sector resolutions and evaluates the retarded Lehmann sum.")
for _binding in ("reaction_energy", "interaction_energy", "energy_difference", "compare",
                 "scan_interaction"):
    _contract(_binding, _binding, "linear", "Returns the specified signed linear combination "
              "of input energies in common units, including counterpoise ghosts where specified.")
for _binding in ("pes_scan", "interaction_scan"):
    _contract(_binding, _binding, "curve", "Returns the ordered tuple of input point energies "
              "at the declared coordinates and fixed geometry/relaxation protocol, not an "
              "interpolated surface or a claim of exact geometry optimization.")
for _binding, _statement in {
    "geometry_transform": "Rigid coordinate change using RDKit-perceived connectivity.",
    "constrained_optimize": "Local constrained geometry and its energy at the declared level.",
    "connected_ts": "Local HF saddle with one imaginary frequency and a connecting IRC; "
                    "this does not certify the global minimum barrier.",
    "scan_fragment": "Specified real/ghost atom component in the declared counterpoise protocol.",
    "neb_path": "Model-defined ASE climbing-image minimum-energy path approximation.",
    "irc": "Model-defined Sella steepest-descent paths and perceived endpoint bond sets.",
    "rate_constant": "Model-defined Eyring transition-state rate with optional Wigner correction.",
}.items():
    _contract(_binding, _binding, "opaque", _statement)
for _binding in ("pauli_shot_estimator", "ideal_energy_projective_sampler"):
    _contract(_binding, "sample", "sample", "Returns a finite-shot estimate of the input "
              "observable in the prepared state, with the reported sampling uncertainty.")
_contract("expectation", "expectation", "expectation", "Returns the exact expectation of "
          "the input observable in the prepared state, not its ground eigenvalue.")
for _binding, _statement in {
    "molecule": "Preserves the specified atoms, coordinate unit, total charge and spin.",
    "geometry": "Preserves the supplied geometry parameters without changing coordinates.",
    "electronic_model": "Constructs the specified basis, reference and electronic model.",
    "scf": "Constructs spin-free spatial integrals in the converged SCF orbital basis.",
    "integrals": "Extracts the input SCF Hamiltonian without changing integrals or constants.",
    "active_space": "Restricts the SCF orbital Hamiltonian to the requested active space, "
                    "including the frozen-core constant in the ledger.",
    "hamiltonian": "Constructs the specified molecular basis/active-space Hamiltonian and "
                   "accounts for nuclear and frozen-core constants once.",
    "sector": "Preserves the declared integer particle counts and optional total-spin target.",
    "encode": "Uses alpha-first Jordan–Wigner parity strings to represent the input "
              "fermionic Hamiltonian, including its constant, on the complete qubit space.",
    "prepare": "Prepares the occupied alpha/beta HF bitstring in the given encoding.",
    "prepare_state": "Builds the requested HF/UCCSD/LUCJ state in the fixed particle sector.",
    "sample_configurations": "Returns computational-basis counts from the prepared state "
                             "using the requested shots and sampling model.",
    "full_subspace": "Enumerates every distinct determinant of the requested particle sector.",
    "subspace_from_samples": "Recovers particle counts and returns the product closure "
                             "of sampled alpha/beta strings as distinct determinants.",
    "extend_subspace": "Applies the specified fermionic ladder and optional excitation "
                       "closure to produce determinants in the destination sector.",
    "lattice_model": "Constructs the declared spin-independent Hubbard hopping and onsite "
                     "interaction integrals in the site basis.",
    "impurity_model": "Constructs the declared Anderson impurity onsite, interaction and "
                      "spin-independent bath-hybridization integrals.",
}.items():
    _contract(_binding, _binding, "data", _statement)
for _binding in ("periodic_scf", "dock", "component_energy"):
    _contract(_binding, _binding, "opaque", "Computes the documented model approximation; "
              "no universal equality or ordering relative to the frozen target is assumed.")
for _binding in ("enumerate_bonds", "homolysis", "boltzmann_weights", "conformer_search",
                 "site_reactivity", "app_component_energy"):
    _contract(_binding, _binding, "opaque", "Evaluates the documented APP C model-defined "
              "quantity. Graph perception, conformer potential and populations are model-defined; "
              "site reactivity is empirical, not an experimental or kinetic prediction.")
for _binding in ("bond_dissociation", "ranking"):
    _contract(_binding, _binding, "energy_collection", "Returns the ordered vector of rigid "
              "fragment sums minus parent energy, or same-formula item energies and their order, "
              "from the specified component inputs; rank pairs use strict summed error bars.")
for _method, _kind in (("hf", "hf"), ("fci", "exact")):
    _name = f"binding:app_component_energy:{_method}"
    AXIOMS[_name] = {"name": _name, "binding": "app_component_energy",
                    "op": "app_component_energy", "kind": _kind, "version": "1",
                    "statement": "Applies the named solver to the exported all-electron "
                    "Hamiltonian and alpha/beta sector, using ROHF spatial orbitals for radicals."}
_contract("thermochemistry", "thermochemistry", "thermal", "Evaluates RRHO ideal-gas "
          "thermal corrections using the input geometry, frequencies, temperature and pressure.")
for _method, _kind in (("fci", "exact"), ("hf", "hf")):
    _name = f"binding:component_energy:{_method}"
    AXIOMS[_name] = {"name": _name, "binding": "component_energy", "op": "component_energy",
                    "kind": _kind, "version": "1", "statement":
                    "Constructs the exported closed-shell molecular Hamiltonian (including "
                    "ghost basis functions) and applies the named FCI or HF solver to it."}
