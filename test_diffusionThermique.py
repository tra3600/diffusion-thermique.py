import numpy as np
import pytest

from diffusionThermique import (differences_finies, nombre_fourier, simuler,
                                solution_exacte, tracer, vecteur_init)

L, D, Tb, Te = 1.0, 0.3, 100.0, 20.0


def test_vecteur_init():
    U = vecteur_init(10, Tb, Te)
    assert U.shape == (11,)
    assert U[0] == U[10] == Te
    assert np.all(U[1:10] == Tb)


def test_vecteur_init_invalide():
    with pytest.raises(ValueError):
        vecteur_init(1, Tb, Te)


def test_conditions_aux_bords_conservees():
    U = vecteur_init(50, Tb, Te)
    V = simuler(U, D, 1e-4, 1 / 50, 200)
    assert V[0] == Te and V[-1] == Te


def test_pas_modifie_entree():
    U = vecteur_init(20, Tb, Te)
    copie = U.copy()
    differences_finies(U, D, 1e-5, 1 / 20)
    np.testing.assert_array_equal(U, copie)


def test_profil_uniforme_stationnaire():
    U = np.full(21, 50.0)
    np.testing.assert_allclose(differences_finies(U, D, 1e-5, 0.05), U)


def test_profil_lineaire_stationnaire():
    U = np.linspace(20, 80, 31)
    np.testing.assert_allclose(differences_finies(U, D, 1e-5, 1 / 30), U)


def test_instabilite_detectee():
    with pytest.raises(ValueError):
        differences_finies(vecteur_init(10, Tb, Te), D, 1.0, 0.1)


def test_nombre_fourier():
    assert nombre_fourier(0.3, 1e-4, 0.01) == pytest.approx(0.3)


def test_principe_du_maximum_et_refroidissement():
    U = vecteur_init(100, Tb, Te)
    V = simuler(U, D, 1e-4, 0.01, 1000)
    assert Te <= V.min() and V.max() <= Tb
    assert V[50] < Tb  # le centre refroidit


def test_convergence_vers_solution_exacte():
    p, n, T = 100, 10000, 1.0
    U = simuler(vecteur_init(p, Tb, Te), D, T / n, L / p, n)
    x = np.linspace(0, L, p + 1)
    exacte = solution_exacte(x, T, L, D, Tb, Te)
    assert np.max(np.abs(U - exacte)) < 1.0


def test_etat_final_tend_vers_Textr():
    p = 20
    U = simuler(vecteur_init(p, Tb, Te), D, 1e-3, 1 / p, 5000)  # r = 0.12
    np.testing.assert_allclose(U, Te, atol=1e-3)


def test_solution_exacte_t0():
    x = np.linspace(0, L, 11)
    T0 = solution_exacte(x, 0.0, L, D, Tb, Te)
    assert T0[0] == Te and T0[5] == Tb


def test_tracer_cree_figure(tmp_path):
    f = tmp_path / "fig.png"
    err = tracer(fichier=str(f), n=2000, p=50)
    assert f.exists() and f.stat().st_size > 0
    assert err < 2.0
