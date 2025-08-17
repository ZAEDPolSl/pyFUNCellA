# GMM Decomposition Script
# This script performs Gaussian Mixture Model decomposition for pathway analysis
#
# Expected inputs:
# - X_data: Single pathway expression data as DataFrame row
# - K_param: Maximum number of components
# - IC_param: Information criterion ("AIC", "AICc", "BIC", "ICL-BIC", "LR")
# - verbose_param: Whether to show verbose output
#
# Output:
# - gmm_results: List containing decomposition results for each pathway

# Load required libraries
library(dpGMM)
library(jsonlite)

# Get parameters
K <- K_param
IC <- IC_param
X <- X_data
verbose_flag <- verbose_param

# GMM options setup
opt <- dpGMM::GMM_1D_opts
opt$max_iter <- 1000
opt$KS <- K
opt$plot <- FALSE
opt$quick_stop <- FALSE
opt$SW <- 0.05
opt$sigmas.dev <- 0
opt$IC <- IC

# Helper function for row calculation that extracts serializable components
row_multiple <- function(row) {
    tmp <- as.numeric(row)
    result <- dpGMM::runGMM(tmp, opts = opt)

    # Extract key components that can be serialized based on runGMM structure
    serializable_result <- list(
        K = result$KS, # Number of components
        IC = result[[IC]], # Information criterion value
        loglik = result$logLik, # Log-likelihood
        threshold = result$threshold, # The thresholds we need!
        cluster = as.vector(result$cluster), # Cluster assignments
        mu = result$model$mu, # Component means
        sigma = result$model$sigma, # Component standard deviations
        alpha = result$model$alpha, # Component weights (not lambda)
        success = TRUE
    )

    # Handle potential NULL values
    if (is.null(serializable_result$mu)) serializable_result$mu <- numeric(0)
    if (is.null(serializable_result$sigma)) serializable_result$sigma <- numeric(0)
    if (is.null(serializable_result$alpha)) serializable_result$alpha <- numeric(0)
    if (is.null(serializable_result$threshold)) serializable_result$threshold <- numeric(0)
    if (is.null(serializable_result$cluster)) serializable_result$cluster <- integer(0)

    serializable_result
}

# GMM Calculation for each row (should be just one row now)
results_list <- list()
total_rows <- nrow(X)

for (i in seq_len(total_rows)) {
    tryCatch(
        {
            row_result <- row_multiple(X[i, ])
            results_list[[rownames(X)[i]]] <- row_result
        },
        error = function(e) {
            if (verbose_flag) {
                cat("Warning: Failed to process pathway", rownames(X)[i], ":", e$message, "\n")
            }
            results_list[[rownames(X)[i]]] <- list(error = e$message, success = FALSE)
        }
    )
}

# Use jsonlite to properly serialize the results
tryCatch(
    {
        # Prepare final results using the expected variable name
        gmm_results <- results_list
    },
    error = function(e) {
        cat("Error serializing results:", e$message, "\n")
        gmm_results <- list(error = "Serialization failed", details = e$message)
    }
)
