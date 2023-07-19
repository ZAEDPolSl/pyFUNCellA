if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!require("GSVA", quietly = TRUE))
    BiocManager::install("GSVA")
if (!require("rjson", quietly = TRUE))
    install.packages("rjson")

library("rjson")
library("GSVA")


args <- commandArgs(trailingOnly=TRUE)
method <- args[1]
pathways <- fromJSON(file="tmp/genesets_genes.json")
gene_expr <- read.csv("tmp/data.csv", row.names = 1, header= TRUE)
res <- gsva(as.matrix(gene_expr), pathways, method = method, kcdf = "Gaussian", mx.diff = F)
write.csv(as.data.frame(res), paste("tmp/res.csv", sep=""))
