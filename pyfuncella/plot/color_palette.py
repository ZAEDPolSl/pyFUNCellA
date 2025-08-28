import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, LinearSegmentedColormap


def get_cluster_palette(n_clusters, clust_sig):
    col_unsig = []
    if clust_sig > 0:
        col_unsig_cmap = LinearSegmentedColormap.from_list(
            "unsig", ["#000000", "#bfbfbf"], N=clust_sig
        )
        col_unsig = [
            to_hex(col_unsig_cmap(i / (clust_sig - 1) if clust_sig > 1 else 0))
            for i in range(clust_sig)
        ]
    col_sig = []
    n_sig = n_clusters - clust_sig
    if n_sig > 0:
        col_sig_cmap = LinearSegmentedColormap.from_list(
            "sig", ["#FFEB6B", "#8B0000"], N=n_sig
        )
        col_sig = [
            to_hex(col_sig_cmap(i / (n_sig - 1) if n_sig > 1 else 0))
            for i in range(n_sig)
        ]
    return col_unsig + col_sig
