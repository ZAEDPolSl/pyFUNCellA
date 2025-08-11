import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm


def plot_pas_distribution(pas_scores, pas_method, pathway_name, threshold, gmm=None):
    """
    Plot PAS score distribution for a pathway, with threshold and optional GMM components.

    Parameters
    ----------
    pas_scores : array-like
        PAS scores for cells.
    pas_method : str
        Name of PAS method (for axis label).
    pathway_name : str
        Name of pathway (for plot title).
    threshold : float
        Threshold value for significance.
    gmm : dict, optional
        GMM parameters from GMMdecomp (dict with keys 'alpha', 'mu', 'sigma').
        If provided, overlays GMM components as dashed lines.

    Notes
    -----
    - For GMM overlay, pass the 'model' dict from GMMdecomp output for the pathway:
      e.g. gmm = results[pathway]['model']
    """

    pas_scores = np.asarray(pas_scores)
    labels = (pas_scores > threshold).astype(int)
    df = pd.DataFrame({"value": pas_scores, "label": labels})

    # Histogram
    hist = go.Histogram(
        x=df["value"],
        marker_color="lightgrey",
        opacity=0.75,
        nbinsx=30,
        showlegend=False,
    )

    # Rugs
    rug0 = go.Scatter(
        x=df[df["label"] == 0]["value"],
        y=[-1] * sum(df["label"] == 0),
        mode="markers",
        marker=dict(color="blue", size=5),
        name="Non significant",
        legendgroup="significance",
        showlegend=True,
        hoverinfo="x",
    )
    rug1 = go.Scatter(
        x=df[df["label"] == 1]["value"],
        y=[-1] * sum(df["label"] == 1),
        mode="markers",
        marker=dict(color="red", size=5),
        name="Significant",
        legendgroup="significance",
        showlegend=True,
        hoverinfo="x",
    )

    fig = go.Figure(data=[hist, rug0, rug1])

    # Histogram stats for scaling GMM
    counts, bin_edges = np.histogram(df["value"], bins=int(np.sqrt(df.shape[0])))
    hist_ymax = counts.max()
    bin_width = bin_edges[1] - bin_edges[0]

    # Overlay GMM if provided (from GMMdecomp)
    if gmm is not None and isinstance(gmm, dict):
        x = np.linspace(df["value"].min() - 1, df["value"].max() + 1, 1000)
        alpha = np.asarray(gmm.get("alpha", []))
        mu = np.asarray(gmm.get("mu", []))
        sigma = np.asarray(gmm.get("sigma", []))
        n_comp = len(alpha)
        for i in range(n_comp):
            pdf = alpha[i] * norm.pdf(x, mu[i], sigma[i])
            scaled_pdf = pdf * len(df) * bin_width
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=scaled_pdf,
                    mode="lines",
                    line=dict(width=2, dash="dot"),
                    name=f"GMM comp. {i+1}",
                    legendgroup="gmm",
                )
            )

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=hist_ymax,
        xref="x",
        yref="y",
        line=dict(color="black", width=2, dash="dash"),
    )

    # Add threshold label
    fig.add_annotation(
        x=threshold,
        y=hist_ymax / 2,
        text=f"thr = {threshold:.2f}",
        showarrow=False,
        ax=2,
        yanchor="bottom",
        font=dict(color="black"),
    )

    # Clean style and adjust legend position
    fig.update_layout(
        title=pathway_name,
        xaxis_title=pas_method,
        yaxis_title="Cell count",
        template="simple_white",
        font=dict(family="Arial", size=14),
        bargap=0.05,
        legend=dict(title="", orientation="h", x=0.5, xanchor="center", y=1.15),
        margin=dict(l=40, r=30, t=50, b=40),
    )

    # Extend y-axis to show rugs
    fig.update_yaxes(range=[-2, None])

    return fig
