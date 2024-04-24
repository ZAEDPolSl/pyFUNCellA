import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import f_oneway, ttest_1samp, tukey_hsd

plot_scorenames = {
    "AUCell": "aucell",
    "CERNO": "auc",
    "JASMINE": "jasmine",
    "DropRatio": "ratios",
    "Mean": "mean",
    "Vision": "vision",
    "Vision abs": "vision_abs",
    "z-score": "z",
    "z-score abs": "z_abs",
    "PLAGE": "svd",
    "PLAGE abs": "svd_abs",
    "SparsePCA": "sparse_pca",
    "SparsePCA abs": "sparse_pca_abs",
    "GSVA": "gsva",
    "ssGSEA": "ssgsea",
    "VAE": "vae",
    "VAE corr": "vae_corr",
}

palette = {
    "AUCell": "#077187",
    "CERNO": "#398D9F",
    "JASMINE": "#6AAAB7",
    "DropRatio": "#586BA4",
    "Mean": "#F4A460",
    "Vision": "#FFC125",
    "Vision abs": "rgba(255,193,37,0.71)",
    "z-score": "#FFD700",
    "z-score abs": "rgba(255,215,0,0.71)",
    "PLAGE": "#A54D69",
    "PLAGE abs": "rgba(165,77,105,0.71)",
    "SparsePCA": "#BC7A8F",
    "SparsePCA abs": "rgba(188,122,143,0.71)",
    "GSVA": "#228B22",
    "ssGSEA": "#006400",
    "VAE": "#000000",
    "VAE corr": "#323232",
}


def check_differences(df, name):
    pval = f_oneway(
        *[list(df[name + sc_name]) for sc_name in list(plot_scorenames.values())]
    ).pvalue
    subtitle = "<br>ANOVA p-value : {}".format(np.round(pval, 3))
    pvals = np.ones(
        (len(list(plot_scorenames.values())), len(list(plot_scorenames.values())))
    )
    if pval < 0.05:
        pvals = tukey_hsd(
            *[list(df[name + sc_name]) for sc_name in list(plot_scorenames.values())]
        ).pvalue
        if pval < 0.001:
            subtitle = "<br>ANOVA p-value < 0.001"
    return pvals, subtitle


def get_brackets(pvals):
    dtype = [("start_idx", int), ("end_idx", int), ("dist", int), ("pvalue", "S10")]
    brackets = np.empty(0, dtype=dtype)
    for i in range(pvals.shape[0]):
        for j in range(i + 1, pvals.shape[1]):
            if pvals[i, j] <= 0.05:
                text = str(np.round(pvals[i, j], 3))
                if pvals[i, j] < 0.001:
                    text = "<0.001"
                bracket = (i, j, j - i, text)
                bracket = np.array(bracket, dtype=dtype)
                brackets = np.append(brackets, bracket)
    brackets = np.sort(brackets, order=["dist", "start_idx"])
    return brackets


def add_brackets(brackets, fig):
    lower = 1.0
    upper = 1.05
    for bracket in brackets:
        i = bracket["start_idx"]
        j = bracket["end_idx"]
        x = (
            [list(plot_scorenames.keys())[i]]
            + list(plot_scorenames.keys())[i:j]
            + [list(plot_scorenames.keys())[j]] * 2
        )
        y = [lower] + [upper] * (j - i + 1) + [lower]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill=None,
                mode="lines",
                line=dict(color="rgba(0,0,0,1)", width=1),
                showlegend=False,
            )
        )
        fig.add_annotation(
            text=str(bracket["pvalue"].decode("UTF-8")),
            name="p-value",
            x=(j + i) / 2,
            y=upper + 0.04,
            showarrow=False,
            font=dict(size=10, color="black"),
        )
        lower = lower + 0.1
        upper = upper + 0.1
    return fig


def add_heatmap(fig, pvals, subtitle):
    fig1 = px.imshow(
        pvals,
        x=[k for k, v in plot_scorenames.items()],
        y=[k for k, v in plot_scorenames.items()],
        text_auto=".2f",
        color_continuous_scale="RdBu",
    )
    z_text = pvals.tolist()
    z_text = [["{:.2f}".format(z) if z >= 0.01 else "<0.01"  for z in z_list] for z_list in z_text]
    fig1.update_traces(
        dict(showscale=False, coloraxis=None, colorscale="RdBu", zmin=0, zmax=1),
        selector={"type": "heatmap"},
    )
    fig1.update_traces(text=z_text, texttemplate="%{text}")

    fig_final = sp.make_subplots(
        rows=2, cols=1, subplot_titles=(subtitle, "Tukey HSD p-values")
    )
    for i, figure in enumerate([fig, fig1]):
        for trace in range(len(figure["data"])):
            fig_final.append_trace(figure["data"][trace], row=i + 1, col=1)
    fig_final.update_layout(height=1200, width=800)
    return fig_final


def mark_different_boxes(fig, pvals, celltype, subtitle):
    if np.sum(pvals <= 0.05) >= 10:
        fig_final = add_heatmap(fig, pvals, subtitle)
    else:
        brackets = get_brackets(pvals)
        fig_final = add_brackets(brackets, fig)
        celltype = celltype + subtitle
    fig_final.update_layout(template="plotly_white", title=celltype)
    return fig_final


def visualize_methods(df, cell_types, namescores, plot_folder):
    for celltype in cell_types:
        for name in namescores:
            vis = df.loc[df["Celltype"].isin([celltype]), df.columns.str.contains(name)]
            pvals, subtitle = check_differences(vis, name)
            vis.columns = vis.columns.str.replace(name, "")
            vis = vis.melt()
            vis = vis.rename(columns={"variable": "method"})
            vis["method"] = vis["method"].map(
                {v: k for k, v in plot_scorenames.items()}
            )
            fig = px.box(
                vis,
                x="method",
                y="value",
                color="method",
                color_discrete_map=palette,
                title=celltype + subtitle,
                template="plotly_white",
                labels={"method": "PAS method", "value": name[:-1].replace("_", " ")},
                category_orders={"method": list(plot_scorenames.keys())},
                height=600,
                width=750,
            )
            fig = mark_different_boxes(fig, pvals, celltype, subtitle)
            fig.update_xaxes(tickangle=45)
            fig.write_image(
                plot_folder + celltype.replace(" ", "_") + "_" + name[:-1] + ".png"
            )


def visualize_difference(df1, df2, namescores, plot_folder, name1, name2):
    df = df1.subtract(df2)
    for name in namescores:
        vis = df.loc[:, df.columns.str.contains(name)]
        vis.columns = vis.columns.str.replace(name, "")
        vis = vis.melt()
        vis = vis.rename(columns={"variable": "method"})
        vis["method"] = vis["method"].map({v: k for k, v in plot_scorenames.items()})
        fig = px.box(
            vis,
            x="method",
            y="value",
            color="method",
            color_discrete_map=palette,
            title=name[:-1].replace("_", " "),
            template="plotly_white",
            labels={
                "method": "PAS method",
                "value": "Difference [{} - {}]".format(name1, name2),
            },
            category_orders={"method": list(plot_scorenames.keys())},
            height=600,
            width=750,
        )
        fig.add_hline(y=0.0, line=dict(dash="dash", color="firebrick", width=1))
        fig.update_xaxes(tickangle=45)
        for score in list(plot_scorenames.keys()):
            vals = vis.loc[vis["method"] == score]
            pval = ttest_1samp(vals["value"], 0.0).pvalue
            if pval < 0.05:
                text = str(np.round(pval, 3))
                if pval < 0.001:
                    text = "<0.001"
                y = vals["value"].min() - 0.05
                if vals["value"].mean() > 0:
                    y = vals["value"].max() + 0.05
                fig.add_annotation(
                    text=text,
                    name="p-value",
                    x=score,
                    y=y,
                    showarrow=False,
                    font=dict(size=10, color="red"),
                )
        if name[:-1] == "FDR":
            name1, name2 = name2, name1
        fig.add_annotation(
            text="Better {}".format(name1),
            name="p-value",
            x=0.5,
            y=vis["value"].max() + 0.15,
            xref="paper",
            showarrow=False,
            font=dict(size=12, color="black"),
        )
        fig.add_annotation(
            text="Better {}".format(name2),
            name="p-value",
            x=0.5,
            y=vis["value"].min() - 0.15,
            xref="paper",
            showarrow=False,
            font=dict(size=12, color="black"),
        )
        fig.write_image(plot_folder + "difference_" + name[:-1] + ".png")
