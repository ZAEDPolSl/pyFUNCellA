import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import numpy as np
import pandas as pd
from .color_palette import get_cluster_palette


def scatterplot_subplots(
    x,
    y,
    continuous_labels=None,
    binary_labels=None,
    ranked_labels=None,
    title="Pathway name",
    pas_method="PAS name",
):
    has_continuous = continuous_labels is not None
    has_binary = binary_labels is not None
    has_ranked = ranked_labels is not None

    plots = []
    if has_continuous:
        plots.append("continuous")
    if has_binary:
        plots.append("binary")
    if has_ranked:
        plots.append("ranked")

    rows = len(plots)
    cols = 1

    title_map = {
        "continuous": "PAS values",
        "binary": "Significance",
        "ranked": "Groups",
    }
    subplot_titles = [title_map[p] for p in plots]
    specs = [[{}] for _ in plots]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
    )
    current_row = 1

    if has_continuous:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                customdata=continuous_labels,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "x: %{x}<br>"
                + "y: %{y}<br>"
                + f"{pas_method}: %{{customdata:.3f}}<br>"
                + "<extra></extra>",
                marker=dict(
                    color=continuous_labels,
                    colorscale="Viridis",
                    size=8,
                    showscale=True,
                    colorbar=dict(
                        title=pas_method,
                        thickness=15,
                        len=0.3,  # Shortened colorbar length
                        y=1.0,
                        yanchor="top",
                        x=1.07,  # Position right side
                    ),
                ),
                showlegend=False,  # No legend entry for continuous; colorbar used
            ),
            row=current_row,
            col=1,
        )
        current_row += 1

    if has_binary:
        binary_df = pd.DataFrame({"x": x, "y": y, "label": binary_labels})
        palette = get_cluster_palette(2, 1)
        for label, color, name in zip(
            [0, 1], palette, ["Non significant", "Significant"]
        ):
            subset = binary_df[binary_df["label"] == label]
            fig.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    marker=dict(color=color, size=8),
                    name=name,
                    legendgroup="binary",
                    showlegend=True,
                ),
                row=current_row,
                col=1,
            )
        current_row += 1

    if has_ranked:
        ranked_df = pd.DataFrame({"x": x, "y": y, "label": ranked_labels})
        unique_ranks = sorted(np.unique(ranked_labels))
        n_clusters = len(unique_ranks)
        if has_binary and binary_labels is not None:
            # A group is significant if any of its binary labels is 1
            unsig_ranks = [
                rank
                for rank in unique_ranks
                if not np.any(np.array(binary_labels)[np.array(ranked_labels) == rank])
            ]
            sig_ranks = [rank for rank in unique_ranks if rank not in unsig_ranks]
            n_unsig = len(unsig_ranks)
            ordered_ranks = unsig_ranks + sig_ranks
        else:
            n_unsig = 1 if n_clusters > 1 else 0
            ordered_ranks = unique_ranks
        palette = get_cluster_palette(n_clusters, n_unsig)
        for i, rank in enumerate(ordered_ranks):
            subset = ranked_df[ranked_df["label"] == rank]
            fig.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    marker=dict(color=palette[i], size=8),
                    name=f"Group {rank}",
                    legendgroup="ranked",
                    showlegend=True,
                ),
                row=current_row,
                col=1,
            )
        current_row += 1

    fig.update_layout(
        height=400 * rows,
        width=750,
        title_text=title,
        title_x=0.02,
        margin=dict(l=80, r=230),  # room on right for colorbar+legend
        legend=dict(
            yanchor="top",
            y=0.65,  # Legend below the colorbar (colorbar y=1 with length=0.3)
            xanchor="left",
            x=1.07,  # Align horizontally with colorbar
            tracegroupgap=350,  # vertical gap between legend groups
            font=dict(size=12),
            bgcolor="rgba(0,0,0,0)",  # transparent background
            borderwidth=0,  # no border
        ),
    )

    return fig
