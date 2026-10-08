# Base-R decoder for the two official fault-free files.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
source_dir <- args[1]
output_dir <- args[2]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
for (kind in c("training", "testing")) {
  title <- if (kind == "training") "Training" else "Testing"
  expected <- paste0("fault_free_", kind)
  env <- new.env()
  loaded <- load(file.path(source_dir, paste0("TEP_FaultFree_", title, ".RData")), envir = env)
  stopifnot(identical(loaded, expected))
  frame <- env[[expected]]
  stopifnot(ncol(frame) == 55, all(vapply(frame, is.numeric, logical(1))))
  stopifnot(identical(names(frame)[1:3], c("faultNumber", "simulationRun", "sample")))
  frame <- frame[order(frame$simulationRun, frame$sample), ]
  mat <- as.matrix(frame)
  stopifnot(all(is.finite(mat)), all(mat[, 1] == 0))
  con <- file(file.path(output_dir, paste0(kind, ".f64")), "wb")
  writeBin(as.double(t(mat)), con, size = 8, endian = "little")
  close(con)
  writeLines(names(frame), file.path(output_dir, paste0(kind, "_columns.txt")))
  indices <- unique(c(1, 2, nrow(mat) %/% 2, nrow(mat) - 1, nrow(mat)))
  check <- cbind(row_zero_based = indices - 1, mat[indices, ])
  write.table(format(check, digits = 17, scientific = TRUE, trim = TRUE),
              file.path(output_dir, paste0(kind, "_spots.csv")),
              sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
  cat(kind, "rows", nrow(mat), "columns", ncol(mat), "runs", length(unique(mat[, 2])), "\n")
  rm(env, frame, mat, check)
  gc()
}
