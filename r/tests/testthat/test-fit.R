set.seed(1)
N <- 50; G <- 300  # 50 cells pad to the 64 bucket; 300 genes are a full chunk plus a short one
X <- cbind(1, rep(0:1, N / 2), rnorm(N)); colnames(X) <- c("a", "b", "c")
off <- log(runif(N, 0.5, 2))
Y <- matrix(rnbinom(G * N, mu = 20, size = 10), G, N, dimnames = list(paste0("g", 1:G), NULL))
ref <- function(family, rows = 1:G, w = rep(1, N)) t(sapply(rows, function(g)
  coef(glm.fit(X, Y[g, ], weights = w, offset = off, family = family, control = glm.control(epsilon = 1e-12)))))

test_that("Poisson fit matches glm.fit", {
  skip_if_no_pjrt()
  fit <- fit_glm(Y, X, offset = off)
  expect_true(all(fit$converged))
  expect_true(is.integer(fit$iterations) && all(fit$iterations > 0))
  expect_equal(unname(fit$Beta), unname(ref(poisson())), tolerance = 1e-7)
  expect_identical(dimnames(fit$Beta), list(rownames(Y), colnames(X)))
  d1 <- glm.fit(X, Y[1, ], offset = off, family = poisson())$deviance
  expect_equal(fit$deviance[1], d1, tolerance = 1e-7)
})

test_that("NB fit matches glm.fit at a fixed dispersion", {
  skip_if_no_pjrt()
  skip_if_not_installed("MASS")
  fit <- fit_glm(Y, X, offset = off, dispersion = 0.1)
  expect_equal(unname(fit$Beta), unname(ref(MASS::negative.binomial(10))), tolerance = 1e-7)
  phi <- rep(c(0.1, 0.5), G / 2)  # per-gene dispersion
  fit2 <- fit_glm(Y, X, offset = off, dispersion = phi)
  expect_equal(unname(fit2$Beta[1:2, ]), rbind(ref(MASS::negative.binomial(10), 1), ref(MASS::negative.binomial(2), 2)),
               tolerance = 1e-7, ignore_attr = TRUE)
})

test_that("weights and sparse input", {
  skip_if_no_pjrt()
  w <- rep(c(1, 0, 2), length.out = N)
  expect_equal(unname(fit_glm(Y[1:5, ], X, offset = off, weights = w)$Beta), unname(ref(poisson(), 1:5, w)),
               tolerance = 1e-7)
  skip_if_not_installed("Matrix")
  S <- Matrix::Matrix(Y[1:5, ], sparse = TRUE)
  expect_equal(fit_glm(S, X, offset = off)$Beta, fit_glm(Y[1:5, ], X, offset = off)$Beta)
})

test_that("single precision runs the f32 kernels, to float32 accuracy", {
  skip_if_no_pjrt()
  fit <- fit_glm(Y, X, offset = off, dispersion = 0.1, precision = "single")
  dbl <- fit_glm(Y, X, offset = off, dispersion = 0.1)
  expect_true(all(fit$converged))
  expect_type(fit$Beta, "double")
  expect_equal(fit$Beta, dbl$Beta, tolerance = 1e-4)
  expect_false(isTRUE(all.equal(fit$Beta, dbl$Beta, tolerance = 1e-12)))  # really float32
  expect_error(fit_glm(Y, X, precision = "half"), "should be one of")
})

test_that("bad input is refused", {
  expect_error(fit_glm(Y, X[-1, ]), "X has 49 rows, Y has 50 columns")
  expect_error(fit_glm(Y, cbind(X, X[, 1])), "not full rank")
  expect_error(fit_glm(Y, matrix(rnorm(N * 13), N)), "at most 12")
  expect_error(fit_glm(-Y, X), "non-negative")
  expect_error(fit_glm(Y, X, dispersion = c(0.1, 0.2)), "one per row")
})
