import numpy as np
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# STABILITÉ AU SENS DU THÉORÈME 6
#
# Le théorème dit : si φ est Lipschitz en y de constante Λ, alors
#
#   max_{0≤n≤N} ||ỹ_n - y_n|| ≤ e^{ΛT} * ( ||ỹ_0 - y_0|| + Σ ||ε_n|| )
#
# On vérifie ceci EMPIRIQUEMENT en :
#   1. Calculant une trajectoire de référence y_n = solve(phi, f, t0, T, x0, h)
#   2. Perturbant la condition initiale : ỹ_0 = x0 + δ
#   3. Calculant ỹ_n = solve(phi, f, t0, T, x0_perturbe, h)
#   4. Mesurant max_n ||ỹ_n - y_n|| et le comparant à la borne e^{ΛT}*||δ||
#
# On fait aussi une variante avec perturbations ε_n à chaque pas.
# ══════════════════════════════════════════════════════════════════════════════

# --- Tes implémentations (déjà définies par toi) ---
# solve(phi, f, t0, T, x0, h)  →  (ts, xs)
# euler, runge_kutta_2, runge_kutta_4  →  fonctions d'incrément φ

import tests_quentin as t

t0, T = 1850.0, 2100.0
h = 0.25



# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Estimer la constante de Lipschitz Λ de φ
#
# φ est Lipschitz en y de constante Λ si :
#   ||φ(t, y1, h) - φ(t, y2, h)|| ≤ Λ * ||y1 - y2||
#
# Pour Euler : φ(t, y, h) = f(t, y), donc Λ_φ = Λ_f = constante de Lipschitz de f
# Pour RK2   : φ mélange f(t,y) et f(t+h, y+h*f), Λ_φ ≈ Λ_f (même ordre)
# Pour RK4   : idem, Λ_φ ≈ Λ_f
#
# On estime Λ_f numériquement via la norme spectrale du Jacobien de f,
# qui est la meilleure constante de Lipschitz locale.
# ══════════════════════════════════════════════════════════════════════════════

def lipschitz_local(f, t, x, eps=1e-5):
    """Estime la constante de Lipschitz de f en (t,x) via ||Jf||_2."""
    n = len(x)
    J = np.zeros((n, n))
    f0 = f(t, x)
    for i in range(n):
        xp = x.copy(); xp[i] += eps
        J[:, i] = (f(t, xp) - f0) / eps
    return np.linalg.norm(J, ord=2)   # norme spectrale = meilleure constante Lipschitz

# On estime Λ sur quelques points de la trajectoire de référence
ts_ref, xs_ref = t.solve(t.euler, t.f, t0, T, t.x0, h)
Lambda_vals = [lipschitz_local(t.f, ts_ref[n], xs_ref[n]) for n in range(0, len(ts_ref), 50)]
Lambda = max(Lambda_vals)   # constante globale (pire cas)

print(f"Constante de Lipschitz Λ estimée : {Lambda:.4f}")
print(f"Borne théorique e^(ΛT) avec T={T-t0} ans : {np.exp(Lambda * (T-t0)):.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : Vérification — perturbation de la condition initiale
#
# On perturbe x0 → x0 + δ et on mesure max_n ||ỹ_n - y_n||
# La borne dit : max_n ||ỹ_n - y_n|| ≤ e^{ΛT} * ||δ||
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)
amplitudes_delta = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]   # ||δ||

methodes = [
    ("Euler (ord. 1)",    t.euler,         "#E24B4A"),
    ("RK2 Heun (ord. 2)", t.runge_kutta_2, "#BA7517"),
    ("RK4 (ord. 4)",      t.runge_kutta_4, "#185FA5"),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Stabilité — Théorème 6 : $\\max_n \\|\\tilde{y}_n - y_n\\| \\leq e^{\\Lambda T} "
    "\\left(\\|\\tilde{y}_0 - y_0\\| + \\sum \\|\\varepsilon_n\\|\\right)$",
    fontsize=12, fontweight="bold"
)

# ── Graphe gauche : erreur max en fonction de ||δ|| ──────────────────────────
ax = axes[0]

for nom, phi, color in methodes:
    _, ys = t.solve(phi, t.f, t0, T, t.x0, h)         # trajectoire non perturbée
    erreurs_max = []

    for amp in amplitudes_delta:
        direction = np.random.randn(len(t.x0))
        direction /= np.linalg.norm(direction)
        delta = amp * direction                   # perturbation de norme amp

        x0_pert = t.x0 + delta
        _, ys_pert = t.solve(phi, t.f, t0, T, x0_pert, h)

        ecarts = [np.linalg.norm(ys_pert[n] - ys[n]) for n in range(len(ys))]
        erreurs_max.append(max(ecarts))

    ax.loglog(amplitudes_delta, erreurs_max,
              'o-', color=color, lw=2, ms=7, label=nom)

# Borne théorique : droite de pente 1 (erreur ∝ ||δ||) avec constante e^{ΛT}
delta_line = np.array([min(amplitudes_delta), max(amplitudes_delta)])
borne = np.exp(Lambda * (T - t0)) * delta_line
ax.loglog(delta_line, borne, 'k--', lw=1.5, label=f"Borne $e^{{\\Lambda T}} \\|\\delta\\|$")

# Droite de pente 1 passant par les courbes (pour vérifier la linéarité)
ax.loglog(delta_line, delta_line, 'k:', lw=1, alpha=0.5, label="Pente 1 (référence)")

ax.set_xlabel(r"$\|\delta\| = \|\tilde{y}_0 - y_0\|$ (amplitude perturbation)", fontsize=10)
ax.set_ylabel(r"$\max_n \|\tilde{y}_n - y_n\|$", fontsize=10)
ax.set_title("Perturbation de la condition initiale", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Vérification — perturbations ε_n à chaque pas
#
# On réimplémente la boucle manuellement pour injecter ε_n à chaque pas :
#   ỹ_{n+1} = ỹ_n + h * φ(t_n, ỹ_n, h) + ε_n
#
# La borne dit : max_n ||ỹ_n - y_n|| ≤ e^{ΛT} * (||δ|| + Σ_n ||ε_n||)
# Ici δ = 0 (même condition initiale), Σ||ε_n|| = N * eps_amp
# ══════════════════════════════════════════════════════════════════════════════

def solve_avec_bruit(phi, f, t0, T, x0, h, eps_amplitude):
    """
    Intègre avec une perturbation aléatoire ε_n à chaque pas.
    Retourne la trajectoire perturbée et la somme ||ε_n||.
    """
    ts = np.arange(t0, T + h, h)
    if ts[-1] > T + h/2: ts = ts[:-1]
    ts[-1] = T

    N = len(ts) - 1
    xs = np.empty((N+1, len(x0)))
    xs[0] = x0.copy()
    somme_eps = 0.0

    for n in range(N):
        h_n = ts[n+1] - ts[n]
        eps = eps_amplitude * np.random.randn(len(x0))   # bruit gaussien
        somme_eps += np.linalg.norm(eps)
        xs[n+1] = xs[n] + h_n * phi(f, ts[n], xs[n], h_n) + eps

    return ts, xs, somme_eps


amplitudes_eps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

ax = axes[1]

for nom, phi, color in methodes:
    _, ys = t.solve(phi, t.f, t0, T, t.x0, h)   # trajectoire sans bruit
    erreurs_max = []
    bornes_th   = []

    for amp in amplitudes_eps:
        np.random.seed(0)
        _, ys_pert, somme_eps = solve_avec_bruit(phi, t.f, t0, T, t.x0, h, amp)

        ecarts = [np.linalg.norm(ys_pert[n] - ys[n]) for n in range(len(ys))]
        erreurs_max.append(max(ecarts))
        # Borne théorique : e^{ΛT} * Σ||ε_n||  (pas de perturbation initiale ici)
        bornes_th.append(np.exp(Lambda * (T - t0)) * somme_eps)

    ax.loglog(amplitudes_eps, erreurs_max,
              'o-', color=color, lw=2, ms=7, label=nom)

# Borne théorique (on la trace pour Euler, elle est la même pour toutes)
ax.loglog(amplitudes_eps, bornes_th,
          'k--', lw=1.5, label=f"Borne $e^{{\\Lambda T}} \\sum\\|\\varepsilon_n\\|$")
ax.loglog(amplitudes_eps, amplitudes_eps,
          'k:', lw=1, alpha=0.5, label="Pente 1 (référence)")

ax.set_xlabel(r"Amplitude des perturbations $\|\varepsilon_n\|$", fontsize=10)
ax.set_ylabel(r"$\max_n \|\tilde{y}_n - y_n\|$", fontsize=10)
ax.set_title("Perturbations à chaque pas de temps", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("stabilite_theoreme6.png", dpi=150, bbox_inches="tight")
plt.show()