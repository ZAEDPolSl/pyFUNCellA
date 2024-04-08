if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "http://cran.us.r-project.org")
BiocManager::install(version = "3.18")
if (!require("GSVA", quietly = TRUE))
    BiocManager::install("GSVA")
if (!require("rjson", quietly = TRUE))
    install.packages("rjson", repos = "http://cran.us.r-project.org")

library("rjson")
library("GSVA")


args <- commandArgs(trailingOnly=TRUE)
method <- args[1]
outpath <- args[2]
pathways <- fromJSON(file=paste(outpath, "tmp/genesets_genes.json", sep=""))
gene_expr <- read.csv(paste(outpath, "tmp/data.csv",  sep=""), row.names=1, header=TRUE, check.names=FALSE)
res <- gene_expr
tryCatch(
{
  res <- gsva(as.matrix(gene_expr), pathways, method = method, kcdf = "Gaussian", mx.diff = F) 
},
error=function(e) {
    conditionMessage(e)
})
write.csv(as.data.frame(res), paste(outpath, "tmp/res.csv", sep=""))
