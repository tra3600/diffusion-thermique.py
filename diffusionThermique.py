import numpy as np

def vecteur_init(p: int, Tbarre: float, Textr: float) -> np.ndarray:
    """
    Initialise le vecteur de température U pour la barre à l'instant initial t0 = 0.
    
    Paramètres:
    p (int): Nombre de subdivisions de la barre.
    Tbarre (float): Température initiale de la barre.
    Textr (float): Température extérieure aux extrémités de la barre.
    
    Retourne:
    np.ndarray: Vecteur de température U.
    """
    U = np.zeros(p + 1)
    U[0] = Textr
    U[p] = Textr
    U[1:p] = Tbarre
    return U

# Exemple d'utilisation
L = 1.0
Tbarre = 100.0
Textr = 20.0
p = 100

U = vecteur_init(p, Tbarre, Textr)
print(U)


def differences_finies(U: np.ndarray, D: float, dt: float, dx: float) -> np.ndarray:
    """
    Calcule la température de la barre à l'instant ti+1 en fonction de la température à l'instant ti.
    
    Paramètres:
    U (np.ndarray): Vecteur de température à l'instant ti.
    D (float): Coefficient de diffusion thermique.
    dt (float): Durée d'un sous-intervalle de temps.
    dx (float): Longueur d'un tronçon de la barre.
    
    Retourne:
    np.ndarray: Vecteur de température à l'instant ti+1.
    """
    p = len(U) - 1
    V = np.zeros(p + 1)
    V[0] = U[0]
    V[p] = U[p]
    for i in range(1, p):
        V[i] = U[i] + D * dt / (dx * dx) * (U[i - 1] - 2 * U[i] + U[i + 1])
    return V

# Exemple d'utilisation
D = 0.3
n = 10000
T = 1.0
dt = T / n
dx = L / p

V = differences_finies(U, D, dt, dx)
print(V)

import numpy as np
import matplotlib.pyplot as plt

# Paramètres globaux
L = 1.0
D = 0.3
T = 1.0
Tbarre = 100.0
Textr = 20.0
n = 10000
p = 100

# Calcul des intervalles de temps et de longueur
dt = T / n
dx = L / p

# Initialisation du vecteur de température
U = vecteur_init(p, Tbarre, Textr)

# Temps pour lequel on souhaite tracer les courbes
temps = np.arange(0, T + 0.1, 0.1)

# Tracé de l'évolution de la température
plt.figure(figsize=(10, 6))
for t in temps:
    plt.plot(np.linspace(0, L, p + 1), U, label=f"t = {t:.1f} s")
    for _ in range(int(0.1 / dt)):
        U = differences_finies(U, D, dt, dx)

plt.xlabel("Position (x)")
plt.ylabel("Température (U)")
plt.title("Évolution de la température de la barre")
plt.legend()
plt.show()