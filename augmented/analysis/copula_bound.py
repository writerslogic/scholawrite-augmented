"""Compute copula-adjusted compound forgery probability, replacing product bound."""

import json
import numpy as np
from scipy.stats import norm, multivariate_normal

# --- Marginal evasion probabilities ---
p_t = 0.818   # temporal (SWF skip detection, k=20, f=0.01)
p_b = 0.002   # behavioral (ROC at τ=0.15)
p_c = 0.10    # content (conservative estimate)

# --- Load measured cross-domain correlations ---
with open("/Volumes/A/researchpapers/analysis/bootstrap_results.json") as f:
    bootstrap = json.load(f)

corr = bootstrap["cross_domain_correlations"]

# Cross-domain mapping:
# entropy ↔ CLC captures temporal-content coupling (ρ=0.071)
# CLC ↔ pause_freq captures content-behavioral coupling (ρ=0.033)
# entropy ↔ iki_log_var is intra-behavioral, not cross-domain
# For temporal-behavioral, use clc_rho_vs_iki_log_var (ρ=0.146) as upper bound
rho_tb = corr["clc_rho_vs_iki_log_var"]["rho"]  # 0.1455
rho_tc = corr["entropy_vs_clc_rho"]["rho"]       # 0.0707
rho_bc = corr["clc_rho_vs_pause_freq"]["rho"]    # 0.0331

print(f"Marginal probabilities:")
print(f"  p_t (temporal)   = {p_t}")
print(f"  p_b (behavioral) = {p_b}")
print(f"  p_c (content)    = {p_c}")
print(f"\nCross-domain correlations (from bootstrap):")
print(f"  ρ(t,b) = {rho_tb:.4f}  [clc_rho ↔ iki_log_var]")
print(f"  ρ(t,c) = {rho_tc:.4f}  [entropy ↔ clc_rho]")
print(f"  ρ(b,c) = {rho_bc:.4f}  [clc_rho ↔ pause_freq]")

# --- Bound 1: Independence (product) ---
P_indep = p_t * p_b * p_c
print(f"\n{'='*60}")
print(f"BOUND 1 — Independence (product):")
print(f"  P_forge = {P_indep:.6e}")

# --- Bound 2: Gaussian copula ---
# Transform to standard normal quantiles
z_t = norm.ppf(p_t)
z_b = norm.ppf(p_b)
z_c = norm.ppf(p_c)
print(f"\nNormal quantiles: z_t={z_t:.4f}, z_b={z_b:.4f}, z_c={z_c:.4f}")

# Correlation matrix
R = np.array([
    [1.0,    rho_tb, rho_tc],
    [rho_tb, 1.0,    rho_bc],
    [rho_tc, rho_bc, 1.0   ]
])

# Verify R is positive definite
eigvals = np.linalg.eigvalsh(R)
print(f"Eigenvalues of R: {eigvals}  (all positive = valid)")

# Joint CDF via multivariate normal
z_vec = np.array([z_t, z_b, z_c])
mvn = multivariate_normal(mean=np.zeros(3), cov=R)
P_copula = mvn.cdf(z_vec)

print(f"\nBOUND 2 — Gaussian copula:")
print(f"  P_forge = {P_copula:.6e}")

# --- Bound 3: Fréchet-Hoeffding worst case ---
P_frechet = min(p_t, p_b, p_c)
print(f"\nBOUND 3 — Fréchet-Hoeffding worst case:")
print(f"  P_forge ≤ {P_frechet:.6e}")

# --- Ratio ---
ratio = P_copula / P_indep
print(f"\n{'='*60}")
print(f"Copula / Independence ratio: {ratio:.4f}")
print(f"  → Copula bound is {ratio:.2f}× the independence bound")

if ratio < 2.0:
    interp = (
        f"The Gaussian copula joint probability ({P_copula:.6e}) is only "
        f"{ratio:.2f}× the independence bound ({P_indep:.6e}). "
        f"With all cross-domain correlations below 0.15, dependence inflates "
        f"the forgery probability by less than a factor of {ratio:.1f}. "
        f"The independence approximation is therefore adequate; "
        f"even the Fréchet worst-case ({P_frechet:.4f}) remains far below "
        f"any practical security threshold."
    )
else:
    interp = (
        f"The copula bound ({P_copula:.6e}) is {ratio:.2f}× the independence "
        f"bound ({P_indep:.6e}), indicating non-trivial dependence. "
        f"The product approximation underestimates the true joint probability."
    )

print(f"\nInterpretation: {interp}")

# --- Save results ---
results = {
    "marginal_probabilities": {"p_t": p_t, "p_b": p_b, "p_c": p_c},
    "normal_quantiles": {"z_t": round(z_t, 6), "z_b": round(z_b, 6), "z_c": round(z_c, 6)},
    "cross_domain_correlations": {
        "t_b": round(rho_tb, 4),
        "t_c": round(rho_tc, 4),
        "b_c": round(rho_bc, 4),
        "sources": {
            "t_b": "clc_rho_vs_iki_log_var",
            "t_c": "entropy_vs_clc_rho",
            "b_c": "clc_rho_vs_pause_freq"
        }
    },
    "correlation_matrix_eigenvalues": [round(e, 6) for e in eigvals],
    "bounds": {
        "independence": round(P_indep, 10),
        "gaussian_copula": round(P_copula, 10),
        "frechet_hoeffding_worst": round(P_frechet, 6)
    },
    "copula_to_independence_ratio": round(ratio, 6),
    "interpretation": interp
}

out_path = "/Volumes/A/researchpapers/analysis/copula_bound_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {out_path}")
