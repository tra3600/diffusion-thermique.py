"""Diffusion thermique 1D dans une barre : équation de la chaleur
dT/dt = D d²T/dx², résolue par différences finies (schéma explicite FTCS)
avec conditions de Dirichlet aux extrémités.
"""
import argparse

import numpy as np


def vecteur_init(p: int, Tbarre: float, Textr: float) -> np.ndarray:
    """Vecteur de température initial (p + 1 points) : Textr aux deux
    extrémités, Tbarre à l'intérieur."""
    if p < 2:
        raise ValueError("p doit être >= 2")
    U = np.full(p + 1, float(Tbarre))
    U[0] = U[p] = Textr
    return U


def nombre_fourier(D: float, dt: float, dx: float) -> float:
    """Nombre r = D dt / dx² ; le schéma explicite est stable si r <= 1/2."""
    return D * dt / (dx * dx)


def differences_finies(U: np.ndarray, D: float, dt: float, dx: float) -> np.ndarray:
    """Un pas de temps du schéma explicite (version vectorisée)."""
    r = nombre_fourier(D, dt, dx)
    if r > 0.5:
        raise ValueError(
            f"Schéma instable : D*dt/dx² = {r:.3f} > 0.5 (réduire dt ou augmenter dx)"
        )
    V = U.copy()
    V[1:-1] = U[1:-1] + r * (U[:-2] - 2 * U[1:-1] + U[2:])
    return V


def simuler(U0: np.ndarray, D: float, dt: float, dx: float, n_pas: int) -> np.ndarray:
    """Avance U0 de n_pas pas de temps et renvoie l'état final."""
    U = U0
    for _ in range(n_pas):
        U = differences_finies(U, D, dt, dx)
    return U


def solution_exacte(x: np.ndarray, t: float, L: float, D: float,
                    Tbarre: float, Textr: float, termes: int = 200) -> np.ndarray:
    """Solution analytique (série de Fourier) pour la condition initiale
    uniforme Tbarre et les bords maintenus à Textr."""
    if t <= 0:
        T = np.full_like(x, Tbarre, dtype=float)
        T[0] = T[-1] = Textr
        return T
    somme = np.zeros_like(x, dtype=float)
    for k in range(1, termes + 1, 2):  # seuls les modes impairs contribuent
        somme += (4 / (k * np.pi)) * np.sin(k * np.pi * x / L) \
            * np.exp(-D * (k * np.pi / L) ** 2 * t)
    return Textr + (Tbarre - Textr) * somme


def tracer(L=1.0, D=0.3, T=1.0, Tbarre=100.0, Textr=20.0, n=10000, p=100,
           fichier="diffusion_thermique.png", afficher=False):
    """Trace l'évolution de la température (numérique en trait plein,
    analytique en pointillés) et la sauvegarde dans `fichier`."""
    import matplotlib
    if not afficher:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dt, dx = T / n, L / p
    x = np.linspace(0, L, p + 1)
    U = vecteur_init(p, Tbarre, Textr)
    pas_par_courbe = n // 10
    instants = [0.0] + [T * (k + 1) / 10 for k in range(10)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    couleurs = plt.cm.plasma(np.linspace(0, 0.9, len(instants)))
    erreur_max = 0.0
    for i, (t, c) in enumerate(zip(instants, couleurs)):
        if i > 0:
            U = simuler(U, D, dt, dx, pas_par_courbe)
        ax1.plot(x, U, color=c, label=f"t = {t:.1f} s")
        ax1.plot(x, solution_exacte(x, t, L, D, Tbarre, Textr), "k:", lw=0.8,
                 label="solution exacte" if i == 0 else None)
        if t > 0:
            erreur_max = max(erreur_max,
                             np.max(np.abs(U - solution_exacte(x, t, L, D, Tbarre, Textr))))
    ax1.set(xlabel="Position x (m)", ylabel="Température (°C)",
            title="Évolution de la température de la barre")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Carte espace-temps
    U = vecteur_init(p, Tbarre, Textr)
    snaps = [U]
    for _ in range(100):
        U = simuler(U, D, dt, dx, n // 100)
        snaps.append(U)
    im = ax2.imshow(np.array(snaps), aspect="auto", origin="lower",
                    extent=[0, L, 0, T], cmap="inferno")
    fig.colorbar(im, ax=ax2, label="Température (°C)")
    ax2.set(xlabel="Position x (m)", ylabel="Temps t (s)",
            title=f"Carte espace-temps (erreur max vs exacte : {erreur_max:.2f} °C)")
    fig.tight_layout()
    fig.savefig(fichier, dpi=130)
    if afficher:
        plt.show()
    plt.close(fig)
    return erreur_max


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", default="diffusion_thermique.png")
    ap.add_argument("--afficher", action="store_true")
    args = ap.parse_args()
    err = tracer(fichier=args.sortie, afficher=args.afficher)
    print(f"Figure enregistrée dans {args.sortie} (erreur max : {err:.3f} °C)")
