import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from enrichment_auc.distributions import categorize_by_thresholds


def clean_up_layout(fig, gs_name, labels_len, embed_name):
    fig.update_layout(
        height=1200,
        width=1000,
        template="plotly_white",
        title_text="Visualizations for {}".format(gs_name),
        legend_tracegroupgap=600 - (labels_len + 1) * 10,
        legend=dict(yanchor="top", y=1.0)
    )
    x_range = fig.full_figure_for_development(warn=False).layout.xaxis.range
    y_range = fig.full_figure_for_development(warn=False).layout.yaxis.range
    for i in range(1, 4):
        fig.update_yaxes(title_text="{} 2".format(embed_name),
                         range=y_range,
                         row=i, col=1)
    fig.update_xaxes(title_text="{} 1".format(embed_name),
                     range=x_range,
                     row=3, col=1)
    return fig


def prepare_subplot(fig, embed, labels, subplot_name, row):
    palette = px.colors.qualitative.Bold
    for i, label in enumerate(np.unique(labels)):
        cell_idx = np.where(labels == label)[0]
        fig.add_trace(
            go.Scatter(
                x=embed[cell_idx, 0],
                y=embed[cell_idx, 1],
                name=str(label),
                marker=dict(color=palette[i]),
                legendgroup=subplot_name,
                legendgrouptitle_text=subplot_name,
                mode="markers",
            ),
            row=row,
            col=1,
        )
    
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return fig


def plot_flow(
    embed, pas, thrs, labels, name, gs_name="", embed_name="", save_dir="plots/"
):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.02
    )
    
    fig = prepare_subplot(fig, embed, labels, "Original", 1)

    fig.add_trace(
        go.Scatter(
            x=embed[:, 0],
            y=embed[:, 1],
            showlegend=False,
            legendgroup="flow",
            name=name,
            marker=dict(
                color=pas,
                colorbar=dict(title=name, len=0.25),
                colorscale="teal"
            ),
            mode="markers",
        ),
        row=2,
        col=1,
    )

    preds_bin = categorize_by_thresholds(pas, thrs).astype(int)
    fig = prepare_subplot(fig, embed, preds_bin, "Clustered", 3)

    labels_len = np.unique(labels).shape[0]
    fig = clean_up_layout(fig, gs_name, labels_len, embed_name)
    fig.write_html(save_dir + "scatter_{}_{}.html".format(gs_name, name))
