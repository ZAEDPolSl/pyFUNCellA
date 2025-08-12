# GSVA/SSGSEA Analysis Script
# This script runs GSVA or SSGSEA analysis using the modern GSVA interface
#
# Expected inputs:
# - gene_expr: Gene expression matrix (genes x samples)
# - genesets: Named list of gene sets
# - method_param: Analysis method ("gsva" or "ssgsea")
#
# Output:
# - gsva_results: List containing scores, samples, and pathways

# Load required libraries
suppressMessages({
    library(methods)
    library(stats)
    library(utils)
    library(GSVA)
})

# Get parameters
gene_expression <- gene_expr
pathways <- genesets
analysis_method <- method_param

# Initialize result variable
gsva_results <- list(error = "Initialization failed")

# Run GSVA/SSGSEA analysis
tryCatch(
    {
        # Use the new GSVA interface with method-specific parameter objects
        if (analysis_method == "ssgsea") {
            # For ssGSEA, use ssgseaParam
            param <- ssgseaParam(
                expr = as.matrix(gene_expression),
                geneSets = pathways
            )
            results <- gsva(param)
        } else if (analysis_method == "gsva") {
            # For GSVA, use gsvaParam
            param <- gsvaParam(
                expr = as.matrix(gene_expression),
                geneSets = pathways,
                kcdf = "Gaussian"
            )
            results <- gsva(param)
        } else {
            stop(paste("Unsupported method:", analysis_method))
        }

        # Convert results matrix to list format for JSON serialization
        # Each row (pathway) becomes a named list element
        results_list <- list()
        for (i in seq_len(nrow(results))) {
            pathway_name <- rownames(results)[i]
            pathway_scores <- as.numeric(results[i, ])
            results_list[[pathway_name]] <- pathway_scores
        }

        # Also include column names for proper reconstruction
        sample_names <- colnames(results)

        # Create the final result structure and assign to the expected variable name
        gsva_results <- list(
            scores = results_list,
            samples = sample_names,
            pathways = rownames(results)
        )
    },
    error = function(e) {
        gsva_results <<- list(error = paste("GSVA error:", e$message))
    }
)

# Ensure gsva_results is the final result variable that gets returned
gsva_results
