# AUCell Threshold Calculation Script
# This script calculates AUCell thresholds for a single pathway
#
# Expected inputs:
# - pathway_scores: Single pathway data as a DataFrame row
#
# Output:
# - aucell_results: Threshold value for the pathway

# Load required libraries
suppressMessages({
    library(AUCell)
})

# Get the single pathway data
pathway_data <- pathway_scores

# Convert to matrix and ensure proper format
pathway_matrix <- as.matrix(pathway_data)

# Use the exact same approach as FUNCellA
# AUCell:::.auc_assignmnetThreshold_v6(as.matrix(df_path[i,]),plotHist = F)$selected
result_obj <- AUCell:::.auc_assignmnetThreshold_v6(pathway_matrix, plotHist = FALSE)

# Extract the selected threshold
if (!is.null(result_obj) && !is.null(result_obj$selected)) {
    aucell_results <- result_obj$selected
} else {
    aucell_results <- result_obj
}
