#' Fit one negative-binomial GLM per gene
#'
#' Fits `Y[g, ] ~ NB(mu, dispersion[g])` with `log(mu) = X %*% beta + offset` for
#' every row of `Y`, at a dispersion you supply. The kernel is compiled for the
#' device in use on first call (see [xlaport::xla_kernel()]) and reused for the
#' rest of the session.
#'
#' Data is padded to the shapes the shipped programs were exported for: cells
#' with weight 0, the design with zero columns. Padding does not change the
#' result. Genes are fitted in chunks, so `Y` may be anything that subsets to
#' rows and converts with `as.matrix()`, such as a sparse `Matrix`.
#'
#' @param Y Counts, genes in rows and cells (samples) in columns.
#' @param X Design matrix, one row per column of `Y`, full rank, at most 12 columns.
#' @param offset Per-cell offset on the log scale, e.g. `log(size_factors)`.
#' @param dispersion NB dispersion, one value or one per gene (variance is
#'   `mu + dispersion * mu^2`). 0, the default, is a Poisson fit.
#' @param weights Per-cell prior weights.
#' @param device Passed to [xlaport::xla_kernel()]; `NULL` uses `PJRT_PLATFORM`.
#' @param precision `"double"` (default) or `"single"`: the float64 or float32
#'   kernels. Single precision is for GPUs, whose float64 rate is a fraction of
#'   their float32 one (1/64 on consumer cards); coefficients then carry float32
#'   rounding (about 1e-6 relative), and results come back as R doubles either way.
#' @return A list: `Beta` (genes x coefficients), `deviance`, `converged` and
#'   `iterations`, one entry per gene. Check `converged` on hard data.
#' @examples
#' \dontrun{
#' Y <- matrix(rpois(2000, 5), 100, 20)
#' X <- cbind(1, rep(0:1, 10))
#' fit <- fit_glm(Y, X, dispersion = 0.1)
#' head(fit$Beta)
#' }
#' @export
fit_glm <- function(Y, X, offset = NULL, dispersion = 0, weights = NULL, device = NULL,
                    precision = c("double", "single")) {
  precision <- match.arg(precision)
  X <- as.matrix(X)
  G <- nrow(Y); N <- ncol(Y); p <- ncol(X)
  if (is.null(offset)) offset <- numeric(N)
  if (is.null(weights)) weights <- rep(1, N)
  if (length(dispersion) == 1) dispersion <- rep(dispersion, G)
  if (is.null(G) || G < 1) stop("Y must be a matrix with at least one row")
  if (nrow(X) != N) stop(sprintf("X has %d rows, Y has %d columns", nrow(X), N))
  if (length(offset) != N || length(weights) != N) stop("offset and weights need one value per column of Y")
  if (length(dispersion) != G) stop("dispersion needs one value, or one per row of Y")
  if (!all(is.finite(X)) || !all(is.finite(offset))) stop("X and offset must be finite")
  if (!all(is.finite(weights)) || any(weights < 0)) stop("weights must be finite and non-negative")
  if (!all(is.finite(dispersion)) || any(dispersion < 0)) stop("dispersion must be finite and non-negative")
  # the kernel ridges X'WX, so a rank-deficient design would get an answer instead of an error
  if (qr(X)$rank < p) stop("X is not full rank")

  # every shape comes from the manifest: Y is [gene chunk, cell bucket], X is [cell bucket, max p]
  ks <- xlaport::xla_kernels(system.file("xla", package = "glmjax"), device)
  dtype <- if (precision == "double") "f64" else "f32"
  ks <- Filter(function(k) identical(attr(k, "spec")$inputs[[1]]$dtype, dtype), ks)
  if (length(ks) == 0) stop("no ", dtype, " kernels in this build of glmjax")
  shape <- function(k, i) as.integer(unlist(attr(k, "spec")$inputs[[i]]$shape))
  cells <- vapply(ks, function(k) shape(k, 1)[2], 0L)
  P <- shape(ks[[1]], 2)[2]
  if (p > P) stop(sprintf("X has %d columns, the kernels support at most %d", p, P))
  if (N > max(cells)) stop(sprintf("Y has %d columns, the kernels support at most %d", N, max(cells)))
  k <- ks[[which(cells == min(cells[cells >= N]))]]
  Gc <- shape(k, 1)[1]; Nb <- shape(k, 1)[2]

  Xp <- matrix(0, Nb, P); Xp[seq_len(N), seq_len(p)] <- X
  off <- c(as.numeric(offset), numeric(Nb - N))
  w <- c(as.numeric(weights), numeric(Nb - N))  # padded cells have weight 0
  Beta <- matrix(NA_real_, G, p, dimnames = list(rownames(Y), colnames(X)))
  deviance <- numeric(G); converged <- logical(G); iterations <- integer(G)
  for (start in seq(1, G, by = Gc)) {
    idx <- start:min(start + Gc - 1, G)
    rows <- rep_len(idx, Gc)  # a short last chunk repeats its genes: no extra iterations, unlike zero rows
    Yc <- as.matrix(Y[rows, , drop = FALSE])
    if (anyNA(Yc) || any(Yc < 0)) stop("Y must be non-negative with no missing values")
    Yp <- matrix(0, Gc, Nb); Yp[, seq_len(N)] <- Yc
    r <- k(Yp, Xp, off, w, as.numeric(dispersion[rows]))
    keep <- seq_along(idx)
    Beta[idx, ] <- r$coef[keep, seq_len(p)]
    deviance[idx] <- r$deviance[keep]
    converged[idx] <- as.logical(r$converged[keep])
    iterations[idx] <- r$n_iter[keep]
  }
  list(Beta = Beta, deviance = deviance, converged = converged, iterations = iterations)
}
