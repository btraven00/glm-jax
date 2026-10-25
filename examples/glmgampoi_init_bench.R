# Why is glmGamPoi slower than edgeR, and does a better initialiser fix it?
#
#   pixi run -e test python examples/sc_bench.py OUT_DIR N_CELLS N_GENES   # or GLMJAX_DATA=real.npz ... 0 0
#   taskset -c 0 Rscript examples/glmgampoi_init_bench.R OUT_DIR LABEL
#
# Compares, on one core and the same data: edgeR::mglmLevenberg (tol 1e-6 / 1e-8), glmGamPoi's
# fitBeta_fisher_scoring from its estimate_betas_roughly start and from a vectorised one-step
# IRLS start (two GEMMs + tiny solves), and glm_gp end to end with the initialiser swapped in.
# Findings (2026-10-25): glmGamPoi is ~1.4x faster per iteration than edgeR but needs 13
# iterations vs 5 because of its unweighted log-OLS start; with the one-step IRLS start it
# converges in 4 and beats edgeR. See ~/phd/worklog/glmgampoi-parity-and-pr-ideas.md.

suppressPackageStartupMessages({library(glmGamPoi); library(edgeR)}); d <- commandArgs(TRUE)[1]; lab <- commandArgs(TRUE)[2]
dims <- scan(file.path(d, "dims.txt"), quiet = TRUE); N <- dims[1]; G <- dims[2]; P <- dims[3]; PHI <- 0.1
rd <- function(k, nr, nc) matrix(readBin(file.path(d, paste0(k, ".bin")), "double", nr * nc), nr, nc)
Y <- rd("Y", G, N); X <- rd("X", N, P); off <- as.vector(rd("offset", N, 1)); ref <- rd("coef", G, P)
off_mat <- matrix(off, G, N, byrow = TRUE); yc <- beachmat::initializeCpp(Y); eoc <- beachmat::initializeCpp(exp(off_mat))
tm <- function(expr) system.time(expr)[["elapsed"]]
out <- function(method, t, iter, B) cat(sprintf("%s,%s,N=%d,p=%d,time=%.2fs,median_iter=%s,time_per_iter=%s,max|b-jax|=%.1e\n", lab, method, N, P, t,
  if (is.na(iter)) "NA" else sprintf("%g", iter), if (is.na(iter)) "NA" else sprintf("%.3fs", t / iter), max(abs(B - ref))))
# vectorised one-step IRLS initialiser (the PR candidate): two GEMMs + G tiny solves, no gene loop
irls1 <- function(Y, X, offset_matrix, theta = 0, pseudo = 0.1) {
  mu0 <- Y + pseudo; W <- mu0 / (1 + theta * mu0); Z <- log(mu0) + (Y - mu0) / mu0 - offset_matrix
  p <- ncol(X); XX <- X[, rep(seq_len(p), each = p)] * X[, rep(seq_len(p), times = p)]
  A <- W %*% XX; b <- (W * Z) %*% X; B <- matrix(0, nrow(Y), p)
  for (g in seq_len(nrow(Y))) B[g, ] <- solve(matrix(A[g, ], p, p), b[g, ])
  B
}
# edgeR kernel (Levenberg-damped Fisher scoring, C), default tol 1e-6 and 1e-8
for (tol in c(1e-6, 1e-8)) { t <- tm(e <- mglmLevenberg(Y, design = X, dispersion = PHI, offset = off, tol = tol)); out(sprintf("edgeR_mglmLevenberg_tol%g", tol), t, median(e$iter), e$coefficients) }
# glmGamPoi kernel from its rough init, tol 1e-8 (default) and 1e-6
rough <- glmGamPoi:::estimate_betas_roughly(Y, X, off_mat)
run <- function(init, tol) glmGamPoi:::fitBeta_fisher_scoring(yc, X, eoc, rep(PHI, G), init, ridge_penalty_nl = NULL, tolerance = tol, max_rel_mu_change = 1e5, max_iter = 1000)
for (tol in c(1e-8, 1e-6)) { t <- tm(r <- run(rough, tol)); out(sprintf("glmGamPoi_kernel_roughinit_tol%g", tol), t, median(r$iter), r$beta_mat) }
t_init <- tm(init1 <- irls1(Y, X, off_mat, PHI)); cat(sprintf("%s,irls1_init_vectorised,N=%d,p=%d,time=%.2fs\n", lab, N, P, t_init))
for (tol in c(1e-8, 1e-6)) { t <- tm(r <- run(init1, tol)); out(sprintf("glmGamPoi_kernel_irls1init_tol%g", tol), t + t_init, median(r$iter), r$beta_mat) }
# end to end: glm_gp with fixed dispersion, stock vs patched initialiser
t0 <- tm(f0 <- glm_gp(Y, design = X, offset = off, size_factors = FALSE, overdispersion = rep(PHI, G), overdispersion_shrinkage = FALSE))
assignInNamespace("estimate_betas_roughly", function(Y, model_matrix, offset_matrix, pseudo_count = 1, ridge_penalty = NULL) irls1(Y, model_matrix, offset_matrix, PHI), ns = "glmGamPoi")
t1 <- tm(f1 <- glm_gp(Y, design = X, offset = off, size_factors = FALSE, overdispersion = rep(PHI, G), overdispersion_shrinkage = FALSE))
cat(sprintf("%s,glm_gp_fixed_disp: stock %.2fs -> irls1 init %.2fs (%.2fx), max|dBeta|=%.1e\n", lab, t0, t1, t0 / t1, max(abs(f0$Beta - f1$Beta))))
t_e <- tm(ge <- glmFit(Y, design = X, dispersion = PHI, offset = off, prior.count = 0)); cat(sprintf("%s,edgeR_glmFit_call: %.2fs\n", lab, t_e))
