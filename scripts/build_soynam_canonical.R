#!/usr/bin/env Rscript
#
# Build the intermediate phenotype/genotype tables for the CRAN SoyNAM
# canonical dataset (Issue #6).
#
# This script is deliberately narrow:
#
#   * it reads the data objects out of a checksum-verified SoyNAM source
#     tarball, without installing or loading the SoyNAM package itself, and
#     without using the NAM package at all. The official BLUP() genotype path
#     (snpQC / markov / duplicate removal) is NOT used: its MAF filter is
#     inert on this data, its marker removal shifts column indices, and its
#     duplicate removal is RNG-driven. Only the phenotype model below is
#     taken from the official implementation.
#   * it fits the phenotype model verbatim from SoyNAM::BLUP()'s
#     use.check = FALSE branch:
#
#         lmer(Y ~ (1 | E) + (1 | G))        # REML, lme4 defaults
#         adjusted = rowMeans(ranef()$G) + mean(Y, na.rm = TRUE)
#
#     use.check = TRUE is not used: in the raw data.line object every line
#     carries the single set code "2A", so the check covariate is missing for
#     the six environments without 2A checks (10,355 observations dropped with
#     no documented intent) and the remaining rows receive another set's check
#     value instead of their own.
#   * it writes plain intermediate files only. The canonical, deterministic
#     outputs are produced by soynam_cran.py from these intermediates.
#
# Usage:
#   Rscript scripts/build_soynam_canonical.R \
#     --tarball=<SoyNAM_1.6.2.tar.gz> --workdir=<dir> --outdir=<dir>

suppressPackageStartupMessages(library(lme4))

EXPECTED <- list(
  r_version = "4.5.3",
  lme4 = "2.0.6",
  matrix = "1.7.5",
  environments = 18L,
  markers = 4312L
)

parse_arguments <- function(argv) {
  wanted <- c("tarball", "workdir", "outdir")
  values <- list()
  for (name in wanted) {
    hit <- grep(paste0("^--", name, "="), argv, value = TRUE)
    if (length(hit) != 1L) {
      stop(sprintf("exactly one --%s=<path> argument is required", name))
    }
    values[[name]] <- sub(paste0("^--", name, "="), "", hit)
  }
  values
}

fail <- function(...) stop(paste0(...), call. = FALSE)

check_environment <- function() {
  actual <- paste(R.version$major, R.version$minor, sep = ".")
  if (actual != EXPECTED$r_version) {
    fail("R ", EXPECTED$r_version, " is required for the canonical dataset; found ", actual)
  }
  for (item in list(
    list(pkg = "lme4", want = EXPECTED$lme4),
    list(pkg = "Matrix", want = EXPECTED$matrix)
  )) {
    have <- as.character(packageVersion(item$pkg))
    if (have != item$want) {
      fail(item$pkg, " ", item$want, " is required; found ", have)
    }
  }
}

load_objects <- function(tarball, workdir) {
  extracted <- file.path(workdir, "source")
  dir.create(extracted, recursive = TRUE, showWarnings = FALSE)
  untar(tarball, exdir = extracted)
  env <- new.env(parent = emptyenv())
  for (name in c("soynam", "soybase")) {
    path <- file.path(extracted, "SoyNAM", "data", paste0(name, ".RData"))
    if (!file.exists(path)) fail("missing data object in tarball: ", name, ".RData")
    load(path, envir = env)
  }
  for (name in c("data.line", "data.check", "gen.qa")) {
    if (!exists(name, envir = env, inherits = FALSE)) {
      fail("tarball did not provide the object '", name, "'")
    }
  }
  env
}

adjusted_phenotypes <- function(data.line) {
  Y <- data.line[, "yield"]
  G <- data.line[, "strain"]
  E <- data.line[, "environ"]
  fit <- lmer(Y ~ (1 | E) + (1 | G))
  if (!isREML(fit)) fail("the phenotype model must be fitted by REML")
  if (isSingular(fit)) fail("the phenotype model produced a singular fit")
  messages <- summary(fit)$optinfo$conv$lme4$messages
  if (!is.null(messages)) {
    fail("lme4 reported convergence problems: ", paste(messages, collapse = " | "))
  }
  adjusted <- rowMeans(ranef(fit)$G) + mean(Y, na.rm = TRUE)
  if (!all(is.finite(adjusted))) fail("adjusted phenotypes must all be finite")
  list(
    values = adjusted,
    fit = fit,
    used_rows = nrow(fit@frame),
    source_rows = nrow(data.line),
    dropped_missing_yield = sum(is.na(Y)),
    environments = length(unique(as.character(E)))
  )
}

family_of_strain <- function(data.line) {
  # The family comes from the phenotype table, never from the strain ID text.
  mapping <- unique(data.frame(
    strain = as.character(data.line$strain),
    family = as.integer(data.line$family),
    stringsAsFactors = FALSE
  ))
  duplicated_strains <- unique(mapping$strain[duplicated(mapping$strain)])
  if (length(duplicated_strains) > 0L) {
    fail(length(duplicated_strains), " strain(s) map to more than one family")
  }
  if (anyNA(mapping$family)) fail("the phenotype table contains rows without a family")
  rownames(mapping) <- mapping$strain
  mapping
}

main <- function() {
  argv <- commandArgs(trailingOnly = TRUE)
  args <- parse_arguments(argv)
  check_environment()

  env <- load_objects(args$tarball, args$workdir)
  data.line <- get("data.line", envir = env)
  gen.qa <- get("gen.qa", envir = env)

  if (ncol(gen.qa) != EXPECTED$markers) {
    fail("gen.qa must carry ", EXPECTED$markers, " markers; found ", ncol(gen.qa))
  }
  if (is.null(rownames(gen.qa)) || is.null(colnames(gen.qa))) {
    fail("gen.qa must carry both sample and marker names")
  }

  phenotype <- adjusted_phenotypes(data.line)
  if (phenotype$environments != EXPECTED$environments) {
    fail(
      "expected ", EXPECTED$environments, " environments; found ",
      phenotype$environments
    )
  }
  mapping <- family_of_strain(data.line)

  joined <- intersect(names(phenotype$values), rownames(gen.qa))
  old_collate <- Sys.getlocale("LC_COLLATE")
  on.exit(Sys.setlocale("LC_COLLATE", old_collate), add = TRUE)
  Sys.setlocale("LC_COLLATE", "C")
  joined <- sort(joined)

  families <- mapping[joined, "family"]
  if (anyNA(families)) fail("joined samples without a family are not allowed")

  genotypes <- gen.qa[joined, , drop = FALSE]
  observed <- genotypes[!is.na(genotypes)]
  if (!all(observed %in% c(0, 1, 2))) {
    fail("gen.qa carries values outside {0, 1, 2, NA}")
  }

  dir.create(args$outdir, recursive = TRUE, showWarnings = FALSE)

  phenotype_table <- data.frame(
    strain = joined,
    family = families,
    value = sprintf("%.17g", phenotype$values[joined]),
    stringsAsFactors = FALSE
  )
  write.table(
    phenotype_table,
    file.path(args$outdir, "phenotype.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, eol = "\n"
  )

  # Written line by line rather than through write.table/data.frame: strain
  # IDs contain characters that data.frame column naming would rewrite.
  transposed <- t(genotypes)
  text_values <- ifelse(is.na(transposed), "NA", as.character(transposed))
  dim(text_values) <- dim(transposed)
  genotype_lines <- c(
    paste(c("marker_id", joined), collapse = "\t"),
    paste(
      colnames(genotypes),
      apply(text_values, 1, paste, collapse = "\t"),
      sep = "\t"
    )
  )
  genotype_con <- file(file.path(args$outdir, "genotype.tsv"), open = "wb")
  writeLines(genotype_lines, genotype_con, sep = "\n")
  close(genotype_con)

  variance <- as.data.frame(VarCorr(phenotype$fit))
  summary_lines <- c(
    paste0("r_version\t", paste(R.version$major, R.version$minor, sep = ".")),
    paste0("lme4_version\t", as.character(packageVersion("lme4"))),
    paste0("matrix_version\t", as.character(packageVersion("Matrix"))),
    paste0("blas\t", basename(sessionInfo()$BLAS %||% "unknown")),
    paste0("lapack\t", basename(sessionInfo()$LAPACK %||% "unknown")),
    paste0("formula\tyield ~ (1 | environ) + (1 | strain)"),
    paste0("reml\tTRUE"),
    paste0("use_check\tFALSE"),
    paste0("source_phenotype_rows\t", phenotype$source_rows),
    paste0("dropped_missing_yield\t", phenotype$dropped_missing_yield),
    paste0("model_frame_rows\t", phenotype$used_rows),
    paste0("environments\t", phenotype$environments),
    paste0("adjusted_strains\t", length(phenotype$values)),
    paste0("joined_samples\t", length(joined)),
    paste0("joined_families\t", length(unique(families))),
    paste0("markers\t", ncol(genotypes)),
    paste0("genotype_missing_rate\t", sprintf("%.17g", mean(is.na(genotypes)))),
    paste0("variance_genotype\t", sprintf("%.17g", variance$vcov[1])),
    paste0("variance_environment\t", sprintf("%.17g", variance$vcov[2])),
    paste0("variance_residual\t", sprintf("%.17g", variance$vcov[3]))
  )
  writeLines(summary_lines, file.path(args$outdir, "build-summary.tsv"), sep = "\n")

  session <- file(file.path(args$outdir, "session-info.txt"), open = "wt")
  writeLines(capture.output(print(sessionInfo())), session)
  close(session)

  cat("wrote intermediates for", length(joined), "samples,",
      length(unique(families)), "families,", ncol(genotypes), "markers\n")
}

`%||%` <- function(x, y) if (is.null(x)) y else x

main()
