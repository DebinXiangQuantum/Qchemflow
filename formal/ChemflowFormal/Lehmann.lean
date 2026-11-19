import ChemflowFormal.Variational
import Mathlib.Tactic.LinearCombination

namespace ChemflowFormal

open scoped InnerProductSpace
open ComplexConjugate

variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E] [Fintype ι]

theorem eigenbasis_coordinates (H : E →ₗ[ℂ] E) (basis : OrthonormalBasis ι ℂ E)
    (energy : ι → ℝ) (heigen : ∀ i, H (basis i) = (energy i : ℂ) • basis i)
    (v : E) (i : ι) : basis.repr (H v) i = (energy i : ℂ) * basis.repr v i := by
  classical
  rw [basis.repr_apply_apply]
  conv_lhs => rw [← basis.sum_repr v]
  simp only [map_sum, map_smul, heigen, inner_sum, inner_smul_right]
  simp [← basis.repr_apply_apply, mul_comm]

/-- The inverse is specified by its unique preimage, not by a numerical algorithm.
Apply this identity to v = c†|0⟩ in the particle sector and v = c|0⟩ in the hole
sector, with the respective shifted Hamiltonians, to obtain the implemented Lehmann sum. -/
theorem lehmann_completeness (H : E →ₗ[ℂ] E) (basis : OrthonormalBasis ι ℂ E)
    (energy : ι → ℝ) (heigen : ∀ i, H (basis i) = (energy i : ℂ) • basis i)
    (z : ℂ) (hz : ∀ i, z - (energy i : ℂ) ≠ 0) (v : E) :
    (∃! u : E, z • u - H u = v) ∧
    ∀ u : E, z • u - H u = v →
      ⟪v, u⟫_ℂ = ∑ i, (Complex.normSq ⟪basis i, v⟫_ℂ : ℂ) / (z - energy i) := by
  classical
  have coord (u : E) (hu : z • u - H u = v) (i : ι) :
      basis.repr u i = basis.repr v i / (z - energy i) := by
    have hc := congrArg (fun w => basis.repr w i) hu
    simp only [map_sub, map_smul, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul,
      eigenbasis_coordinates H basis energy heigen] at hc
    apply (eq_div_iff (hz i)).mpr
    linear_combination hc
  constructor
  · let u := ∑ i, (basis.repr v i / (z - energy i)) • basis i
    have hu : z • u - H u = v := by
      apply basis.repr.injective
      ext i
      simp only [map_sub, map_smul, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul,
        eigenbasis_coordinates H basis energy heigen]
      have hc : basis.repr u i = basis.repr v i / (z - energy i) := by
        rw [basis.repr_apply_apply]
        dsimp [u]
        simp only [inner_sum, inner_smul_right]
        simp [← basis.repr_apply_apply]
      rw [hc]
      field_simp [hz i]
      ring
    refine ⟨u, hu, ?_⟩
    intro w hw
    apply basis.repr.injective
    ext i
    rw [coord w hw, coord u hu]
  · intro u hu
    rw [← basis.sum_inner_mul_inner v u]
    apply Finset.sum_congr rfl
    intro i _
    rw [← basis.repr_apply_apply, coord u hu, basis.repr_apply_apply]
    rw [Complex.normSq_eq_conj_mul_self, inner_conj_symm]
    ring

omit [Fintype ι] in
theorem upper_half_plane_avoids_eigenvalues (energy : ι → ℝ) (z : ℂ) (hz : 0 < z.im) :
    ∀ i, z - (energy i : ℂ) ≠ 0 := by
  intro i heq
  have him := congrArg Complex.im heq
  simp only [Complex.sub_im, Complex.ofReal_im, sub_zero, Complex.zero_im] at him
  linarith

end ChemflowFormal
