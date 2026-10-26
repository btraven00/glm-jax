# glmGamPoi README benchmark (pbmc4k), run against one glmGamPoi build.
#
#   R_LIBS=<lib with glmGamPoi build>:$R_LIBS_USER taskset -c N Rscript glmgampoi_readme_bench.R H5 LABEL [refs]
#
# H5 is TENxPBMCData's pbmc4k_rectangular.h5 (ExperimentHub EH1611). Reproduces the
# README's bench::mark (300 random non-empty genes, intercept-only design, in memory and
# on disk) and adds a non-group design (intercept + 2-level factor + continuous covariate),
# which glm_gp fits by Fisher scoring rather than group-wise means, plus a 2000-gene
# version. "refs" also times edgeR and DESeq2 as in the README. One CSV row per timing.

suppressPackageStartupMessages({ library(glmGamPoi); library(HDF5Array) })
a <- commandArgs(TRUE); h5 <- a[1]; lab <- a[2]; refs <- isTRUE(a[3] == "refs")
counts <- HDF5Array(h5, "counts")
set.seed(1)
non_empty <- which(DelayedMatrixStats::rowSums2(counts) > 0)
idx2000 <- sample(non_empty, 2000)
sets <- list(g300 = idx2000[1:300], g2000 = idx2000)
n <- ncol(counts)
designs <- list(intercept = matrix(1, nrow = n),
                group_cont = cbind(1, rep(0:1, length.out = n), rnorm(n)))

row <- function(genes, design, method, b) {
  t <- as.numeric(b$median); itr <- length(b$time[[1]])
  cat(sprintf("%s,%s,%s,%s,%.3f,%d\n", lab, genes, design, method, t, itr))
}
for (gs in names(sets)) {
  Y <- as.matrix(counts[sets[[gs]], ])
  for (ds in names(designs)) {
    X <- designs[[ds]]
    row(gs, ds, "glmGamPoi_in_memory", bench::mark(glm_gp(Y, design = X, on_disk = FALSE), min_iterations = 3, check = FALSE))
    row(gs, ds, "glmGamPoi_on_disk", bench::mark(glm_gp(Y, design = X, on_disk = TRUE), min_iterations = 3, check = FALSE))
    # isolate the beta fit: dispersions fixed to the per-gene estimates of a first fit
    disp <- glm_gp(Y, design = X, on_disk = FALSE)$overdispersions
    row(gs, ds, "glmGamPoi_fixed_disp", bench::mark(glm_gp(Y, design = X, on_disk = FALSE, overdispersion = disp,
                                                           overdispersion_shrinkage = FALSE), min_iterations = 3, check = FALSE))
    if (refs) {
      row(gs, ds, "edgeR", bench::mark({
        d <- edgeR::DGEList(Y); d <- edgeR::calcNormFactors(d)
        d <- edgeR::estimateDisp(d, X); edgeR::glmFit(d, design = X)
      }, min_iterations = 3, check = FALSE))
      if (gs == "g300" && ds == "intercept") row(gs, ds, "DESeq2", bench::mark(suppressMessages({  # as in the README
        dds <- DESeq2::DESeqDataSetFromMatrix(Y, colData = data.frame(name = seq_len(n)), design = ~ 1)
        dds <- DESeq2::estimateSizeFactors(dds, "poscounts")
        dds <- DESeq2::estimateDispersions(dds, quiet = TRUE)
        dds <- DESeq2::nbinomWaldTest(dds, minmu = 1e-6)
      }), min_iterations = 3, check = FALSE))
    }
  }
}
# Same fits from both builds? Print a fingerprint to compare across runs
f <- glm_gp(as.matrix(counts[sets$g2000, ]), design = designs$group_cont, on_disk = FALSE)
cat(sprintf("%s,fingerprint,sum_deviance=%.6f,sum_overdisp=%.6f,beta2_median=%.6f\n", lab,
            sum(f$deviances), sum(f$overdispersions), median(f$Beta[, 2])))
