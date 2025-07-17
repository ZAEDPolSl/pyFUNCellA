import numpy as np
import pandas as pd

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    RPY2_AVAILABLE = True
except ImportError:
    RPY2_AVAILABLE = False


def thr_AUCell(df_path, pathway_names=None, sample_names=None):
    """
    Calculate activity thresholds for pathways using AUCell's thresholding algorithm.

    Parameters
    ----------
    df_path : pd.DataFrame or np.ndarray
        Pathway activity scores (rows: pathways, columns: samples).
    pathway_names : list, optional
        Names for pathways (rows). Used if df_path is numpy array.
    sample_names : list, optional
        Names for samples (columns). Used if df_path is numpy array.

    Returns
    -------
    dict
        Dictionary mapping pathway names to threshold activity score.
    """
    if not RPY2_AVAILABLE:
        raise ImportError("rpy2 is required for AUCell thresholding.")
    # Convert input to DataFrame if needed
    if isinstance(df_path, np.ndarray):
        n_pathways, n_samples = df_path.shape
        if pathway_names is None:
            pathway_names = [f"pathway_{i}" for i in range(n_pathways)]
        if sample_names is None:
            sample_names = [f"sample_{i}" for i in range(n_samples)]
        df = pd.DataFrame(df_path, index=pathway_names, columns=sample_names)
    else:
        df = df_path.copy()
        if df.index is None or not hasattr(df.index, "size") or df.index.size == 0:
            df = df.copy()
            df.index = pd.Index([f"pathway_{i}" for i in range(df.shape[0])])

    # Define R function for AUCell thresholding (single pathway)
    r_code = """
    auc_threshold_single <- function(scores) {
        suppressMessages({
            library(AUCell)
        })
        res <- AUCell:::.auc_assignmnetThreshold_v6(as.matrix(scores), plotHist=FALSE)
        return(res$selected)
    }
    """
    robjects.r(r_code)
    r_auc_threshold_single = robjects.globalenv["auc_threshold_single"]

    thresholds = {}
    with localconverter(robjects.default_converter + pandas2ri.converter):
        for pathway in df.index:
            # Pass as matrix (1 row, n_samples columns)
            scores_matrix = pd.DataFrame([df.loc[pathway].values], columns=df.columns)
            r_matrix = robjects.conversion.py2rpy(scores_matrix)
            thr = r_auc_threshold_single(r_matrix)[0]
            thresholds[pathway] = thr
    return thresholds
