import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from scipy import optimize, stats

import glm_jax as gj


def nb_data(G=50, n=12, seed=0, phi=0.1):
    rng = np.random.default_rng(seed)
    group = np.repeat([0, 1], n // 2)
    X = np.column_stack([np.ones(n), group, rng.normal(size=n)])
    lib = rng.uniform(0.5, 2.0, size=n) * 1e6
    offset = np.log(lib)
    beta = np.column_stack([
        rng.normal(-11, 1, G), rng.normal(0, 1, G), rng.normal(0, 0.3, G)
    ])
    mu = np.exp(beta @ X.T + offset)
    r = 1 / phi
    Y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    return Y, X, offset, beta


def score(family, y, X, offset, w, phi, beta):
    eta = X @ beta + offset
    mu = family.link.inverse(eta)
    d = family.link.mu_eta(eta)
    return X.T @ (w * (y - mu) * d / family.variance(mu, phi))


@pytest.mark.parametrize("family", [gj.Poisson(), gj.NegativeBinomial()])
def test_score_equations_zero(family):
    Y, X, offset, _ = nb_data()
    phi = np.full(Y.shape[0], 0.1)
    fit = gj.fit_glm(Y, X, family=family, offset=offset, phi=phi)
    assert bool(np.all(fit.converged))
    for g in range(Y.shape[0]):
        s = score(family, Y[g], X, offset, 1.0, phi[g], np.asarray(fit.coef[g]))
        assert np.max(np.abs(s)) < 1e-5


def test_nb_matches_scipy_mle():
    Y, X, offset, _ = nb_data(G=5)
    phi = 0.1
    fam = gj.NegativeBinomial()
    fit = gj.fit_glm(Y, X, family=fam, offset=offset, phi=phi)
    r = 1 / phi
    for g in range(Y.shape[0]):
        def nll(b):
            mu = np.exp(X @ b + offset)
            return -stats.nbinom.logpmf(Y[g], r, r / (r + mu)).sum()
        opt = optimize.minimize(nll, np.asarray(fit.coef[g]) + 0.1, method="BFGS",
                                options={"gtol": 1e-10})
        np.testing.assert_allclose(fit.coef[g], opt.x, atol=1e-4)
        # loglik consistency
        ll = np.sum(fam.loglik(Y[g], fit.mu[g], phi))
        np.testing.assert_allclose(-ll, nll(np.asarray(fit.coef[g])), rtol=1e-10)


def test_gaussian_equals_ols():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(20), rng.normal(size=(20, 2))])
    Y = rng.normal(size=(4, 20))
    fit = gj.fit_glm(Y, X, family=gj.Gaussian())
    ols = np.linalg.lstsq(X, Y.T, rcond=None)[0].T
    np.testing.assert_allclose(fit.coef, ols, atol=1e-8)


def test_binomial_logistic():
    rng = np.random.default_rng(2)
    n = 40
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    trials = rng.integers(5, 30, size=n).astype(float)
    p = 1 / (1 + np.exp(-(0.3 + 0.8 * X[:, 1])))
    succ = rng.binomial(trials.astype(int), p)
    fit = gj.fit_glm(succ / trials, X, family=gj.Binomial(), weights=trials)
    def nll(b):
        return -stats.binom.logpmf(succ, trials, 1 / (1 + np.exp(-(X @ b)))).sum()
    opt = optimize.minimize(nll, np.zeros(2), method="BFGS")
    np.testing.assert_allclose(fit.coef, opt.x, atol=1e-4)


def test_intercept_only_closed_form():
    Y, X, offset, _ = nb_data()
    fit = gj.fit_glm(Y, X[:, :1], family=gj.Poisson(), offset=offset)
    expected = np.log(Y.sum(1) / np.exp(offset).sum())
    np.testing.assert_allclose(fit.coef[:, 0], expected, atol=1e-8)


def test_zero_rows_do_not_crash():
    Y, X, offset, _ = nb_data(G=4)
    Y[0] = 0
    fit = gj.fit_glm(Y, X, offset=offset, phi=0.1)
    assert np.all(np.isfinite(fit.deviance))
    assert fit.deviance[0] < 1e-4


def test_lrt_and_wald():
    Y, X, offset, beta = nb_data(G=200, n=20, phi=0.05)
    test, full, red = gj.glm_lrt(Y, X, gj.drop_coef(X, 1), offset=offset, phi=0.05)
    assert test.df == 1
    assert np.all((test.pvalue >= 0) & (test.pvalue <= 1))
    w = gj.wald_test(full, [0, 1, 0])
    # LRT and Wald should be highly concordant
    assert np.corrcoef(np.asarray(test.stat), np.asarray(w.stat) ** 2)[0, 1] > 0.95


def test_dispersion_grid_recovers_truth():
    Y, X, offset, _ = nb_data(G=300, n=20, phi=0.2, seed=3)
    est = gj.estimate_dispersion_grid(Y, X, offset=offset)
    assert 0.12 < float(est["common"]) < 0.3


def test_warm_start_same_answer():
    Y, X, offset, _ = nb_data(G=10)
    a = gj.fit_glm(Y, X, offset=offset, phi=0.1)
    b = gj.fit_glm(Y, X, offset=offset, phi=0.1, beta0=a.coef)
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-7)
    assert np.all(b.n_iter <= 2)


def test_batch_size_same_answer():
    Y, X, offset, _ = nb_data(G=53)  # 53 not divisible by 16: exercises the remainder chunk
    a = gj.fit_glm(Y, X, offset=offset, phi=0.1)
    b = gj.fit_glm(Y, X, offset=offset, phi=0.1, batch_size=16)
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-12)
    np.testing.assert_array_equal(a.converged, b.converged)


def test_float32_inputs_fit_in_float32():
    Y, X, offset, _ = nb_data(G=20)
    a = gj.fit_glm(Y, X, offset=offset, phi=0.1)
    b = gj.fit_glm(Y.astype(np.float32), X.astype(np.float32), offset=offset, phi=0.1, tol=1e-6, beta_tol=1e-6)
    assert b.coef.dtype == np.float32
    np.testing.assert_allclose(a.coef, b.coef, atol=1e-3)


def test_apl_grid_matches_independent_refits():
    Y, X, offset, _ = nb_data(G=40, n=12, phi=0.2, seed=5)
    grid = np.logspace(-2, 0.5, 6)
    ref = np.stack([gj.adjusted_profile_loglik(Y, X, g, offset=offset) for g in grid], axis=1)
    apl = gj.apl_grid(Y, X, grid, offset=offset, steps=None)
    np.testing.assert_allclose(apl, ref, rtol=1e-6, atol=1e-5)
    # default: 3 fixed warm-started steps per grid point; close, same argmax
    fast = gj.apl_grid(Y, X, grid, offset=offset)
    assert np.all(np.isfinite(fast))
    assert np.max(np.abs(fast - ref)) < 0.05
    np.testing.assert_array_equal(np.argmax(fast, axis=1), np.argmax(ref, axis=1))
    # portable export: no LAPACK custom calls in the StableHLO
    txt = jax.jit(lambda Y, X, g, o: gj.apl_grid(Y, X, g, offset=o)).lower(Y, X, grid, offset).as_text()
    assert "custom_call" not in txt


def test_dispersion_newton_polish_beats_coarse_grid():
    Y, X, offset, _ = nb_data(G=200, n=20, phi=0.2, seed=3)
    fine_grid = np.logspace(-2, 0.5, 400)
    fine = gj.apl_grid(Y, X, fine_grid, offset=offset, steps=None)
    log_phi_fine = np.log(fine_grid[np.argmax(fine, axis=1)])
    coarse = gj.estimate_dispersion_grid(Y, X, offset=offset, newton=0)
    polished = gj.estimate_dispersion_grid(Y, X, offset=offset)  # newton=2
    err_coarse = np.abs(np.log(coarse["tagwise"]) - log_phi_fine)
    err_pol = np.abs(np.log(polished["tagwise"]) - log_phi_fine)
    assert np.all(np.isfinite(polished["tagwise"]))
    assert np.median(err_pol) < 0.5 * np.median(err_coarse)
    assert 0.12 < float(polished["common"]) < 0.3
    # polish stays within one grid spacing of its start
    spacing = np.max(np.diff(np.log(np.asarray(coarse["phi_grid"]))))
    assert np.all(np.abs(np.log(polished["tagwise"]) - np.log(coarse["tagwise"])) <= 2 * spacing + 1e-9)


def test_two_pass_straggler_fit_matches_single_gene_fits():
    # G > _STRAGGLER_BUCKET triggers the capped first pass + bucket refit
    Y, X, offset, _ = nb_data(G=300, n=12, seed=7)
    res = gj.fit_glm(Y, X, offset=offset, phi=0.1)
    assert np.all(res.converged)
    for g in (0, 17, 299):
        one = gj.fit_glm(Y[g], X, offset=offset, phi=0.1)
        np.testing.assert_allclose(res.coef[g], one.coef, atol=1e-7)
