#!/usr/bin/env Rscript

args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
      stop("Usage: ./format_bugs.R output_file_name", call.=FALSE)
}
library(coda)
library(R2OpenBUGS)
coda.data <- as.matrix(read.bugs("CODAchain1.txt"))
param_file <- c()
for (out_param in colnames(as.matrix(coda.data))) {
    stan_out_param = gsub("\\]", "", gsub("\\[", "_", gsub(",", "_", gsub("\\.", "_", out_param))))
    param_file <- cbind(param_file, paste(stan_out_param, ",[", paste(coda.data[, out_param], collapse = " "), "]"))
}
write.table(param_file, file = args[1], row.names=FALSE, col.names=FALSE, sep="\n", quote = FALSE)
