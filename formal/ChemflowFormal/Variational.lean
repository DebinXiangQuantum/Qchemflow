import Mathlib.Analysis.InnerProductSpace.Spectrum
import Mathlib.Analysis.InnerProductSpace.Rayleigh
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace ChemflowFormal

open scoped InnerProductSpace
open Module.End

variable {𝕜 E : Type*} [RCLike 𝕜] [NormedAddCommGroup E]
  [InnerProductSpace 𝕜 E] [FiniteDimensional 𝕜 E]

noncomputable def rayleigh (T : E →ₗ[𝕜] E) (x : E) : ℝ :=
  RCLike.re ⟪T x, x⟫_𝕜 / ‖x‖ ^ 2

noncomputable def ground (T : E →ₗ[𝕜] E) : ℝ :=
  ⨅ x : {x : E // x ≠ 0}, rayleigh T x

theorem rayleigh_bddBelow (T : E →ₗ[𝕜] E) :
    BddBelow (Set.range fun x : {x : E // x ≠ 0} => rayleigh T x) := by
  refine ⟨-‖T.toContinuousLinearMap‖, ?_⟩
  rintro _ ⟨x, rfl⟩
  apply (le_div_iff₀ (sq_pos_of_ne_zero (norm_ne_zero_iff.mpr x.property))).mpr
  have hinner := re_inner_le_norm (𝕜 := 𝕜) (-(T x)) (x : E)
  have hop := T.toContinuousLinearMap.le_opNorm (x : E)
  simp only [inner_neg_left, map_neg, norm_neg] at hinner
  have hmul := mul_le_mul_of_nonneg_right hop (norm_nonneg (x : E))
  change ‖T x‖ * ‖(x : E)‖ ≤ _ at hmul
  dsimp [rayleigh]
  nlinarith

theorem variational_principle [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) (x : E) (hx : x ≠ 0) :
    HasEigenvalue T (ground T : 𝕜) ∧ ground T ≤ rayleigh T x := by
  exact ⟨hT.hasEigenvalue_iInf_of_finiteDimensional,
    ciInf_le (rayleigh_bddBelow T) ⟨x, hx⟩⟩

theorem exact_diagonalization [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) :
    HasEigenvalue T (ground T : 𝕜) ∧
      ∀ μ : ℝ, HasEigenvalue T (μ : 𝕜) → ground T ≤ μ := by
  refine ⟨hT.hasEigenvalue_iInf_of_finiteDimensional, ?_⟩
  intro μ hμ
  obtain ⟨x, hx⟩ := hμ.exists_hasEigenvector
  have heq := mem_eigenspace_iff.mp hx.1
  have hr : rayleigh T x = μ := by
    unfold rayleigh
    rw [heq, inner_smul_left, RCLike.conj_ofReal, RCLike.re_ofReal_mul,
      inner_self_eq_norm_sq_to_K, ← RCLike.ofReal_pow, RCLike.ofReal_re]
    exact mul_div_cancel_right₀ μ (pow_ne_zero 2 (norm_ne_zero_iff.mpr hx.2))
  simpa [hr] using (variational_principle T hT x hx.2).2

noncomputable def ritz (T : E →ₗ[𝕜] E) (V : Submodule 𝕜 E) : ℝ :=
  ⨅ x : {x : V // x ≠ 0}, rayleigh T (x : V)

theorem ritz_lower_bound [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) (V : Submodule 𝕜 E) [Nontrivial V] :
    ground T ≤ ritz T V := by
  obtain ⟨v, hv⟩ := exists_ne (0 : V)
  haveI : Nonempty {x : V // x ≠ 0} := ⟨⟨v, hv⟩⟩
  apply le_ciInf
  intro x
  exact (variational_principle T hT (x : V)
    (fun h => x.property (Subtype.ext h))).2

theorem rayleigh_ritz_full [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) : ritz T ⊤ = ground T := by
  obtain ⟨v, hv⟩ := exists_ne (0 : E)
  haveI : Nonempty {x : E // x ≠ 0} := ⟨⟨v, hv⟩⟩
  haveI : Nontrivial (⊤ : Submodule 𝕜 E) :=
    ⟨⟨⟨v, Submodule.mem_top⟩, 0, fun h => hv (congrArg Subtype.val h)⟩⟩
  apply le_antisymm
  · apply le_ciInf
    intro x
    have hb : BddBelow (Set.range fun y : {y : (⊤ : Submodule 𝕜 E) // y ≠ 0} =>
        rayleigh T (y : (⊤ : Submodule 𝕜 E))) := by
      refine ⟨ground T, ?_⟩
      rintro _ ⟨y, rfl⟩
      exact (variational_principle T hT (y : (⊤ : Submodule 𝕜 E))
        (fun h => y.property (Subtype.ext h))).2
    exact ciInf_le hb ⟨⟨x, Submodule.mem_top⟩, fun h => x.property (congrArg Subtype.val h)⟩
  · exact ritz_lower_bound T hT ⊤

noncomputable def compression (T : E →ₗ[𝕜] E) (V : Submodule 𝕜 E) : V →ₗ[𝕜] V :=
  (orthogonalProjection V).toLinearMap.comp (T.comp V.subtype)

theorem compression_symmetric (T : E →ₗ[𝕜] E) (hT : T.IsSymmetric)
    (V : Submodule 𝕜 E) : (compression T V).IsSymmetric := by
  intro x y
  change ⟪orthogonalProjection V (T x), y⟫_𝕜 = ⟪x, orthogonalProjection V (T y)⟫_𝕜
  rw [inner_orthogonalProjection_eq_of_mem_right, inner_orthogonalProjection_eq_of_mem_left]
  exact hT x y

theorem compression_ground (T : E →ₗ[𝕜] E) (V : Submodule 𝕜 E) :
    ground (compression T V) = ritz T V := by
  unfold ground ritz
  apply iInf_congr
  intro x
  unfold rayleigh
  change RCLike.re ⟪orthogonalProjection V (T (x : V)), (x : V)⟫_𝕜 / ‖(x : V)‖ ^ 2 = _
  rw [inner_orthogonalProjection_eq_of_mem_right]
  rfl

theorem rayleigh_ritz [Nontrivial E] (T : E →ₗ[𝕜] E)
    (hT : T.IsSymmetric) (V : Submodule 𝕜 E) [Nontrivial V] :
    ground T ≤ ritz T V ∧
      HasEigenvalue (compression T V) (ritz T V : 𝕜) ∧
      ∀ μ : ℝ, HasEigenvalue (compression T V) (μ : 𝕜) → ritz T V ≤ μ := by
  refine ⟨ritz_lower_bound T hT V, ?_⟩
  simpa [compression_ground] using exact_diagonalization (compression T V)
    (compression_symmetric T hT V)

omit [FiniteDimensional 𝕜 E] in
theorem sector_decomposition (T N : E →ₗ[𝕜] E) (h : T * N = N * T)
    (μ : 𝕜) (x : E) (hx : x ∈ eigenspace N μ) : T x ∈ eigenspace N μ := by
  rw [mem_eigenspace_iff] at hx ⊢
  have hc := congrArg (fun f : E →ₗ[𝕜] E => f x) h
  change T (N x) = N (T x) at hc
  rw [← hc, hx, map_smul]

theorem sector_variational [Nontrivial E] (T : E →ₗ[𝕜] E) (hT : T.IsSymmetric)
    (V : Submodule 𝕜 E) [Nontrivial V] (hV : ∀ v ∈ V, T v ∈ V)
    (x : V) (hx : x ≠ 0) :
    ground (T.restrict hV) ≤ rayleigh (T.restrict hV) x :=
  (variational_principle (T.restrict hV) (hT.restrict_invariant hV) x hx).2

theorem positive_linear_combination {ι : Type*} (terms : Finset ι)
    (c x target : ι → ℝ)
    (h : ∀ i ∈ terms, x i = target i ∨ (0 ≤ c i ∧ target i ≤ x i)) :
    ∑ i ∈ terms, c i * target i ≤ ∑ i ∈ terms, c i * x i := by
  apply Finset.sum_le_sum
  intro i hi
  rcases h i hi with he | ⟨hc, hx⟩
  · rw [he]
  · exact mul_le_mul_of_nonneg_left hx hc

theorem linear_combination_exact {ι : Type*} (terms : Finset ι)
    (c x target : ι → ℝ) (h : ∀ i ∈ terms, c i ≠ 0 → x i = target i) :
    ∑ i ∈ terms, c i * x i = ∑ i ∈ terms, c i * target i := by
  apply Finset.sum_congr rfl
  intro i hi
  by_cases hc : c i = 0
  · simp [hc]
  · rw [h i hi hc]

end ChemflowFormal
