# Decode the fixed faulty-testing panel.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
source <- args[1]; ids <- scan(args[2],quiet=TRUE); out <- args[3]
stopifnot(length(ids)==50, length(unique(ids))==50)
dir.create(out,recursive=TRUE,showWarnings=FALSE)
env <- new.env(); loaded <- load(source,envir=env)
stopifnot(length(loaded)==1)
frame <- env[[loaded[1]]]
stopifnot(is.data.frame(frame),nrow(frame)==9600000,ncol(frame)==55)
stopifnot(identical(names(frame)[1:3],c('faultNumber','simulationRun','sample')))
stopifnot(all(vapply(frame,is.numeric,logical(1))))
for (name in names(frame)) stopifnot(all(is.finite(frame[[name]])))
stopifnot(all(frame$faultNumber %in% 1:20), all(frame$simulationRun %in% 1:500), all(frame$sample %in% 1:960))
# Count every fault/run; selected records are then checked for exact sample order.
code <- as.integer((frame$faultNumber-1)*500+frame$simulationRun)
stopifnot(all(tabulate(code,nbins=10000)==960))
selected <- frame[frame$simulationRun %in% ids, ]
rm(frame,env,code);gc()
selected <- selected[order(selected$faultNumber,selected$simulationRun,selected$sample), ]
stopifnot(nrow(selected)==960000)
con <- file(file.path(out,'selected.f64'),'wb')
manifest <- vector('list',1000)
for (i in seq_len(1000)) {
  block <- as.matrix(selected[((i-1)*960+1):(i*960), ])
  stopifnot(all(block[,1]==block[1,1]),all(block[,2]==block[1,2]),identical(as.integer(block[,3]),1:960))
  writeBin(as.double(t(block)),con,size=8,endian='little')
  manifest[[i]] <- data.frame(block_zero=i-1,fault=block[1,1],run=block[1,2],rows=960)
}
close(con)
write.csv(do.call(rbind,manifest),file.path(out,'selected_runs.csv'),row.names=FALSE)
writeLines(names(selected),file.path(out,'columns.txt'))
writeLines(c(paste('object',loaded[1]),'source_rows 9600000','source_run_counts_checked 10000',
             'selected_rows 960000','selected_runs 1000','all_source_values_finite TRUE'),file.path(out,'structure.txt'))
cat('decoded1000 predetermined runs; no detector computation\n')
