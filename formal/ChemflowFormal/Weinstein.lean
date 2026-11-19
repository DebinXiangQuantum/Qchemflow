import ChemflowFormal.Lehmann

namespace ChemflowFormal

open scoped InnerProductSpace
open Module.End

variable {E ι : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]

theorem parseval [Fintype ι] (basis : OrthonormalBasis ι ℂ E) (v : E) :
    ∑ i, ‖basis.repr v i‖ ^ 2 = ‖v‖ ^ 2 := by
  rw [← PiLp.norm_sq_eq_of_L2]
  simp

theorem eigenvalue_near_residual [Fintype ι] (H : E →ₗ[ℂ] E)
    (basis : OrthonormalBasis ι ℂ E) (energy : ι → ℝ)
    (heigen : ∀ i, H (basis i) = (energy i : ℂ) • basis i)
    (ψ : E) (hψ : ‖ψ‖ = 1) (μ : ℝ) :
    ∃ i, |energy i - μ| ≤ ‖H ψ - (μ : ℂ) • ψ‖ := by
  have hweight : ∑ i, ‖basis.repr ψ i‖ ^ 2 = 1 := by rw [parseval, hψ]; norm_num
  have hres : ∑ i, (energy i - μ) ^ 2 * ‖basis.repr ψ i‖ ^ 2 =
      ‖H ψ - (μ : ℂ) • ψ‖ ^ 2 := by
    rw [← parseval basis]
    apply Finset.sum_congr rfl
    intro i _
    have hc : basis.repr (H ψ - (μ : ℂ) • ψ) i =
        ((energy i - μ : ℝ) : ℂ) * basis.repr ψ i := by
      simp only [map_sub, map_smul, PiLp.sub_apply, PiLp.smul_apply, smul_eq_mul,
        eigenbasis_coordinates H basis energy heigen, Complex.ofReal_sub]
      ring
    rw [hc, norm_mul, mul_pow, Complex.norm_real, Real.norm_eq_abs, sq_abs]
  by_contra h
  push_neg at h
  have hsquare (i : ι) : ‖H ψ - (μ : ℂ) • ψ‖ ^ 2 < (energy i - μ) ^ 2 := by
    have habs := sq_abs (energy i - μ)
    have hpos := norm_nonneg (H ψ - (μ : ℂ) • ψ)
    nlinarith [h i]
  have hpositive : ∃ i ∈ Finset.univ, 0 < ‖basis.repr ψ i‖ ^ 2 := by
    by_contra hn
    push_neg at hn
    have hzero := Finset.sum_nonpos hn
    rw [hweight] at hzero
    norm_num at hzero
  have hstrict : ∑ i, ‖H ψ - (μ : ℂ) • ψ‖ ^ 2 * ‖basis.repr ψ i‖ ^ 2 <
      ∑ i, (energy i - μ) ^ 2 * ‖basis.repr ψ i‖ ^ 2 := by
    apply Finset.sum_lt_sum
    · intro i _
      exact mul_le_mul_of_nonneg_right (hsquare i).le (sq_nonneg _)
    · obtain ⟨i, hi, hp⟩ := hpositive
      exact ⟨i, hi, mul_lt_mul_of_pos_right (hsquare i) hp⟩
  rw [← Finset.mul_sum, hweight, mul_one, hres] at hstrict
  exact (lt_irrefl _ hstrict)

theorem weinstein_bound [FiniteDimensional ℂ E] (H : E →ₗ[ℂ] E)
    (hH : H.IsSymmetric) (ψ : E) (hψ : ‖ψ‖ = 1) :
    let μ := Complex.re ⟪H ψ, ψ⟫_ℂ
    ∃ eigenvalue : ℝ, HasEigenvalue H (eigenvalue : ℂ) ∧
      |eigenvalue - μ| ≤ Real.sqrt (‖H ψ‖ ^ 2 - μ ^ 2) := by
  dsimp
  let μ := Complex.re ⟪H ψ, ψ⟫_ℂ
  let basis := hH.eigenvectorBasis rfl
  let energy := hH.eigenvalues (n := Module.finrank ℂ E) rfl
  obtain ⟨i, hi⟩ := eigenvalue_near_residual H basis energy
    (hH.apply_eigenvectorBasis rfl) ψ hψ μ
  refine ⟨energy i, hH.hasEigenvalue_eigenvalues rfl i, ?_⟩
  have hv : ‖H ψ - (μ : ℂ) • ψ‖ ^ 2 = ‖H ψ‖ ^ 2 - μ ^ 2 := by
    rw [norm_sub_sq (𝕜 := ℂ), inner_smul_right]
    change ‖H ψ‖ ^ 2 - 2 * ((μ : ℂ) * ⟪H ψ, ψ⟫_ℂ).re +
      ‖(μ : ℂ) • ψ‖ ^ 2 = _
    rw [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero,
      norm_smul, hψ, mul_one,
      Complex.norm_real, Real.norm_eq_abs, sq_abs]
    dsimp [μ]
    ring
  have hs : Real.sqrt (‖H ψ‖ ^ 2 - μ ^ 2) = ‖H ψ - (μ : ℂ) • ψ‖ := by
    rw [← hv, Real.sqrt_sq (norm_nonneg _)]
  exact hs.symm ▸ hi

end ChemflowFormal
