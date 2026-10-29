# Single-cell NB fit benchmark, R side: glm-jax via pjrt vs glmGamPoi vs glm loop.
#
#   Rscript examples/sc_bench.R OUT_DIR
#
# Run examples/sc_bench.py into OUT_DIR first. Prints one CSV row per method:
# method,N,G,compile_s,fit_s,max_abs_coef_diff_vs_jax,median_iter.

suppressPackageStartupMessages({
  library(glmGamPoi)
  library(MASS)
})
HAVE_PJRT <- requireNamespace("pjrt", quietly = TRUE)  # pjrt needs protoc to build; optional

d <- commandArgs(TRUE)[1]
dims <- scan(file.path(d, "dims.txt"), quiet = TRUE)  # N G p, written by sc_bench.py
N <- dims[1]; G <- dims[2]; P <- dims[3]
DT <- Sys.getenv("GLMJAX_DTYPE", "f64")  # must match the dtype sc_bench.py exported with
R_CORES <- as.integer(Sys.getenv("GLMJAX_R_CORES", "1"))  # >1: only the multi-process glmGamPoi row
PHI <- 0.1; N_LOOP <- 100  # genes for the glm loop, extrapolated to G

rd <- function(k, nr, nc) matrix(readBin(file.path(d, paste0(k, ".bin")), "double", nr * nc), nr, nc)
Y <- rd("Y", G, N); X <- rd("X", N, P); off <- as.vector(rd("offset", N, 1)); ref <- rd("coef", G, P)
row <- function(m, comp, fit, diff, iter = NA) cat(sprintf("%s,%d,%d,%.3f,%.3f,%.2e,%s\n", m, N, G, comp, fit, diff, iter))
med3 <- function(f) median(replicate(3, system.time(f())[["elapsed"]]))

# glmGamPoi is single-threaded C++; its multi-core number is gene blocks across forked workers
if (R_CORES > 1) {
  t_par <- system.time(parts <- parallel::mclapply(parallel::splitIndices(G, R_CORES), function(idx)
    glm_gp(Y[idx, , drop = FALSE], design = X, offset = off, size_factors = FALSE,
           overdispersion = rep(PHI, length(idx)), overdispersion_shrinkage = FALSE)$Beta,
    mc.cores = R_CORES))[["elapsed"]]
  row(sprintf("glmGamPoi_%dproc", R_CORES), 0, t_par, max(abs(do.call(rbind, parts) - ref)))
  quit(save = "no")
}

# glm-jax kernel through pjrt
if (HAVE_PJRT) {
  library(pjrt)
  t_comp <- system.time(exe <- pjrt_compile(pjrt_program(path = file.path(d, "fit.mlir"), format = "mlir")))[["elapsed"]]
  bufs <- list(pjrt_buffer(Y, dtype = DT), pjrt_buffer(X, dtype = DT),
               pjrt_buffer(off, dtype = DT), pjrt_buffer(rep(PHI, G), dtype = DT))
  run <- function() as_array(do.call(pjrt_execute, c(list(exe), bufs))[[1]])
  coef <- run()
  row(paste0("pjrt_", DT), t_comp, med3(run), max(abs(coef - ref)))
}

# glmGamPoi at the same fixed dispersion: the user-facing call...
t_ggp <- system.time(fit <- glm_gp(Y, design = X, offset = off, size_factors = FALSE,
                                   overdispersion = rep(PHI, G), overdispersion_shrinkage = FALSE))[["elapsed"]]
row("glmGamPoi", 0, t_ggp, max(abs(fit$Beta - ref)))

# ...and its C++ Fisher-scoring kernel alone (inputs and starting betas prepared outside the timer)
off_mat <- matrix(off, G, N, byrow = TRUE)
init <- glmGamPoi:::estimate_betas_roughly(Y, X, off_mat)
yc <- beachmat::initializeCpp(Y); eoc <- beachmat::initializeCpp(exp(off_mat))
t_k <- system.time(k <- glmGamPoi:::fitBeta_fisher_scoring(yc, X, eoc, rep(PHI, G), init, ridge_penalty_nl = NULL,
                                                           tolerance = 1e-8, max_rel_mu_change = 1e5,
                                                           max_iter = 1000))[["elapsed"]]
row("glmGamPoi_kernel", 0, t_k, max(abs(k$beta_mat - ref)), median(k$iter))

# edgeR glmFit (Levenberg-damped IRLS in C, single-threaded) at the same fixed dispersion
if (requireNamespace("edgeR", quietly = TRUE)) {
  t_e <- system.time(e <- edgeR::glmFit(Y, design = X, dispersion = PHI, offset = off, prior.count = 0))[["elapsed"]]
  row("edgeR_glmFit", 0, t_e, max(abs(e$coefficients - ref)))
}

# DEVIL (caravagnalab/devil) at the same fixed dispersion: init_overdispersion = PHI with
# overdispersion = FALSE fits beta at PHI and skips the dispersion step. Offset goes in as
# size_factors = exp(off - mean(off)): its `offset` argument is rep()'d to n, so only a scalar
# works, and raw exp(off) (~5000) gives NaN for 93% of genes (CPU) / all (CUDA) on Hao 10k.
# The centring moves the intercept by mean(off); the diff is taken against the shifted ref.
# tol 1e-3 is its default; 1e-8 matches the other rows. CPU genes run serially (its default
# BiocParallel backend takes every core). CUDA row only if built --with-cuda, and it is NOT
# at PHI: beta_fit_gpu takes no dispersion and estimates its own jointly with beta, so its
# coefficient diff mixes in a different dispersion.
if (requireNamespace("devil", quietly = TRUE)) {
  cuda <- c(FALSE, if (exists("beta_fit_gpu", envir = asNamespace("devil"))) TRUE)
  for (gpu in cuda) for (tol in c(1e-3, 1e-8)) {
    t_d <- system.time(dv <- devil::fit_devil(Y, X, overdispersion = FALSE, init_overdispersion = PHI,
                                              size_factors = exp(off - mean(off)), tolerance = tol, CUDA = gpu,
                                              BPPARAM = BiocParallel::SerialParam(), verbose = FALSE))[["elapsed"]]
    ref_c <- ref; ref_c[, 1] <- ref[, 1] + mean(off)
    row(sprintf("devil%s_tol%g", if (gpu) "_cuda" else "", tol), 0, t_d, max(abs(dv$beta - ref_c)),
        median(dv$iterations$beta_iters))
  }
}

# naive per-gene glm, timed on N_LOOP genes and scaled to G
idx <- seq_len(min(N_LOOP, G))
t_loop <- system.time(b <- t(sapply(idx, function(g)
  coef(glm.fit(X, Y[g, ], offset = off, family = negative.binomial(1 / PHI))))))[["elapsed"]]
row("glm_loop_extrap", 0, t_loop * G / length(idx), max(abs(b - ref[idx, ])))
