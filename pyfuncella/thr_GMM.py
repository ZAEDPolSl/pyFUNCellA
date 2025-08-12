import numpy as np
from typing import Optional, Callable
from pyfuncella.utils.kmeans_search import km_search
from pyfuncella.utils.progress_callbacks import get_progress_callback


def thr_GMM(gmms, progress_callback: Optional[Callable] = None):
    """
    Calculate thresholds for pathway activity based on GMMs.

    Parameters
    ----------
    gmms : dict
        Dictionary of GMM results, one per pathway. Each value must have:
        - 'thresholds': numeric vector of GMM thresholds
        - 'model': dict with 'mu', 'sigma', 'alpha' (component params)
    progress_callback : callable, optional
        Optional callback function for progress reporting.
        Should accept (current, total, message) parameters.

    Returns
    -------
    dict
        Dictionary with keys as pathway names, values as dicts:
        - 'Kmeans_thr': threshold chosen via k-means (or fallback)
        - 'Top1_thr': maximum original threshold
        - 'All_thr': all thresholds (after NA removal)
    """
    results = {}
    pathway_names = list(gmms.keys())
    total_pathways = len(pathway_names)

    progress_cb = get_progress_callback(
        progress_callback,
        description="Calculating thresholds",
        unit="pathway",
        verbose=True,
    )

    for i, name in enumerate(pathway_names):
        progress_cb(i + 1, total_pathways, f"threshold {name}")
        results[name] = _process_pathway_threshold(gmms[name])

    return results


def _process_pathway_threshold(gmm):
    """Process threshold calculation for a single pathway."""
    # Handle both old format (thresholds array) and new format (single threshold)
    thrs = gmm.get("thresholds")  # Old format for backward compatibility
    single_thr = gmm.get("threshold")  # New format from GMMdecomp

    if thrs is not None:
        # Old format: multiple thresholds as array
        thrs = np.asarray(thrs, dtype=float)
        valid_mask = ~np.isnan(thrs)
        thrs_clean = thrs[valid_mask]
    elif single_thr is not None:
        # New format: single threshold from GMMdecomp
        if np.isnan(single_thr) or np.isinf(single_thr):
            thrs_clean = np.array([])
        else:
            thrs_clean = np.array([float(single_thr)])
    else:
        # No thresholds available
        thrs_clean = np.array([])

    model = gmm.get("model", {})

    # Top1 threshold: max of cleaned thresholds, fallback to nan if empty
    Top1_thr = np.nanmax(thrs_clean) if thrs_clean.size > 0 else float("nan")
    All_thr = thrs_clean.tolist()

    # Prepare params for k-means threshold
    mu = np.asarray(model.get("mu", []))
    sigma = np.asarray(model.get("sigma", []))
    alpha = np.asarray(model.get("alpha", []))

    # Kmeans threshold logic
    if thrs_clean.size > 0:
        n_components = mu.size
        if thrs_clean.size == 1 and n_components == 2:
            Kmeans_thr = thrs_clean[0]
        elif thrs_clean.size != n_components - 1:
            Kmeans_thr = Top1_thr
        else:
            # Build a GMM result dict for km_search
            gmm_for_kmeans = {
                "model": {"mu": mu, "sigma": sigma, "alpha": alpha},
                "thresholds": thrs_clean,
            }
            Kmeans_thr = km_search(gmm_for_kmeans)
    else:
        Kmeans_thr = float("-inf")

    return {
        "Kmeans_thr": Kmeans_thr,
        "Top1_thr": Top1_thr,
        "All_thr": All_thr,
    }
