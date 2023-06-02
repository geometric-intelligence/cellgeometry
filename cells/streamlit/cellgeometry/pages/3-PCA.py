import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import R2, DiscreteCurves, ClosedDiscreteCurves

from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.learning.mdm import RiemannianMinimumDistanceToMean
from geomstats.learning.pca import TangentPCA
from sklearn.decomposition import PCA

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import experimental


st.sidebar.header("STEP 3: PCA")

st.write("# Principal Component Analysis (PCA) 👋")

cells = st.session_state["cells"]
cell_shapes = st.session_state["cell_shapes"]
if cells.size == 0:
    st.warning(
        "👈 Have you uploaded a zipped file of ROIs under Load Data? Afterwards, go the the Cell Shear page and run the analysis there."
    )
    st.stop()


cells_flat = gs.reshape(cells, (len(cells), -1))

R1 = Euclidean(dim=1)
CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)
CURVES_SPACE = DiscreteCurves(R2)
SRV_METRIC = CURVES_SPACE.srv_metric
L2_METRIC = CURVES_SPACE.l2_curves_metric

n_components = st.slider("Select the Number of Sampling Points", 0, len(cells_flat), 10)

pcas = {}
pcas["Linear"] = PCA(n_components=n_components).fit(cells_flat)
pcas["Linear_components"] = PCA(n_components=n_components).fit_transform(cells_flat)
pcas["SRV"] = TangentPCA(n_components=n_components, metric=SRV_METRIC).fit(cell_shapes)
pcas["SRV_components"] = TangentPCA(
    n_components=n_components, metric=SRV_METRIC
).fit_transform(cell_shapes)

# fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# for i, metric_name in enumerate(["Linear", "SRV"]):
#     axs[i].plot(pcas[metric_name].explained_variance_ratio_)
#     axs[i].set_xlabel("Number of PCS")
#     axs[i].set_ylabel("Explained variance")
#     tangent = ""
#     if metric_name == "SRV":
#         tangent = "Tangent "
#     first_pc_explains = 100*sum(pcas[metric_name].explained_variance_ratio_[:1])
#     axs[i].set_title(f"{tangent}PCA with {metric_name} metric\n 1st PC explains: {first_pc_explains:.1f}%")


st.write("## PCA Analysis")


# Extract principal component values
pc1 = pcas["Linear_components"][:, 0]
pc2 = pcas["Linear_components"][:, 1]
pc3 = pcas["Linear_components"][:, 2]

# Create a 3D scatter trace for the principal components
trace = go.Scatter3d(
    x=pc1,
    y=pc2,
    z=pc3,
    mode="markers",
    marker=dict(
        size=5,  # Adjust marker size as per preference
        color=pc1,  # Use PC1 values for color mapping
        colorscale="Viridis",  # Choose a desired color scale
        opacity=0.8,
        colorbar=dict(title="PC1"),  # Add colorbar with PC1 label
    ),
)

# Create the layout for the plot
layout = go.Layout(
    title="PCA Plot",
    scene=dict(
        xaxis=dict(title="PC1"),
        yaxis=dict(title="PC2"),
        zaxis=dict(title="PC3"),
    ),
)

# Create the Figure object and add the trace to it
fig = go.Figure(data=trace, layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


# Extract principal component values
spc1 = pcas["SRV_components"][:, 0]
spc2 = pcas["SRV_components"][:, 1]
spc3 = pcas["SRV_components"][:, 2]

# Create a 3D scatter trace for the principal components
trace = go.Scatter3d(
    x=spc1,
    y=spc2,
    z=spc3,
    mode="markers",
    marker=dict(
        size=5,  # Adjust marker size as per preference
        color=spc1,  # Use PC1 values for color mapping
        colorscale="Viridis",  # Choose a desired color scale
        opacity=0.8,
        colorbar=dict(title="PC1"),  # Add colorbar with PC1 label
    ),
)

# Create the layout for the plot
layout = go.Layout(
    title="Tangent PCA Plot",
    scene=dict(
        xaxis=dict(title="PC1"),
        yaxis=dict(title="PC2"),
        zaxis=dict(title="PC3"),
    ),
)

# Create the Figure object and add the trace to it
fig = go.Figure(data=trace, layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


st.write("## Explained Variance Ratio")


# Create a subplot with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

for i, metric_name in enumerate(["Linear", "SRV"]):
    tangent = "Tangent " if metric_name == "SRV" else ""
    first_pc_explains = 100 * sum(pcas[metric_name].explained_variance_ratio_[:1])
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pcas[metric_name].explained_variance_ratio_))),
            y=pcas[metric_name].explained_variance_ratio_,
            mode="lines",
            name=f"{metric_name}",
            legendgroup=f"{metric_name}",  # Legend grouping
            hovertemplate="Number of PCS: %{x}<br>Explained Variance: %{y:.3f}",  # Hover text
        ),
        row=1,
        col=i + 1,
    )

    fig.update_xaxes(title_text="Number of PCS", row=1, col=i + 1)
    fig.update_yaxes(title_text="Explained variance", row=1, col=1)

    # Update the subplot title
    fig.update_layout(
        annotations=[
            go.layout.Annotation(
                x=0.5,
                y=-0.2,
                showarrow=False,
                text=f"{tangent}PCA with {metric_name} metric\n 1st PC explains: {first_pc_explains:.1f}%",
                xref="paper",
                yref="paper",
                xanchor="center",
                yanchor="top",
                xshift=0,
                yshift=0,
                font=dict(size=12),
            )
        ],
        legend=dict(y=0.5, traceorder="reversed", font=dict(size=10), yanchor="middle"),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="#fafafa",
    )

# fig.show()
st.plotly_chart(fig)

# Explained Variances
lin_explained = sum(pcas["Linear"].explained_variance_ratio_[:2])
srv_explained = sum(pcas["SRV"].explained_variance_ratio_[:2])


diff_explained = lin_explained - srv_explained

col1, col2 = st.columns(2)
col1.metric("Linear", round(lin_explained, 2), round(diff_explained, 2))
col2.metric("SRV", round(srv_explained, 2), round(-diff_explained, 2))

st.markdown(
    """
__Explained Variance Ratio__: The proportion of the total variance in the data that is explained by _each principal component_.
It quantifies the contribution of each principal component in capturing the underlying patterns and variability present in the dataset.
__Higher values__ indicate that the corresponding principal component explains a larger portion of the variance, while __lower values__ indicate a smaller contribution.
The explained variance ratio provides insights into the relative importance of different components in the dimensionality reduction process.
"""
)

# Get the explained variance ratio of the principal components
explained_variance_ratio = pcas["Linear"].explained_variance_ratio_

# Get the feature weights (loadings) of the original features on the principal components
feature_weights = pcas["Linear"].components_.T

# Create a scatter trace for the cell data
trace_scatter = go.Scatter(
    x=pc1,
    y=pc2,
    mode="markers",
    marker=dict(
        size=5,  # Adjust marker size as per preference
        color="rgb(31, 119, 180)",  # Custom marker color
        opacity=0.8,
    ),
)

# Create arrow annotations for the feature weights
annotations = []
for i in range(len(feature_weights)):
    annotations.append(
        go.layout.Annotation(
            x=0,
            y=0,  # Starting point of the arrow
            xref="x",
            yref="y",
            ax=feature_weights[i, 0],
            ay=feature_weights[i, 1],  # Arrow direction and magnitude
            # text='Feature ' + str(i + 1),  # Label the arrow with feature number
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
        )
    )

# Create the layout for the biplot
layout = go.Layout(
    title="PCA Biplot",
    xaxis=dict(
        title="PC1 ({}% explained variance)".format(
            round(explained_variance_ratio[0] * 100, 2)
        )
    ),
    yaxis=dict(
        title="PC2 ({}% explained variance)".format(
            round(explained_variance_ratio[1] * 100, 2)
        )
    ),
    showlegend=False,
    annotations=annotations,
)

# Create the Figure object and add the scatter trace and annotations to it
fig = go.Figure(data=[trace_scatter], layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


# Extract principal component values
pc1 = pcas["SRV_components"][:, 0]
pc2 = pcas["SRV_components"][:, 1]

# Get the explained variance ratio of the principal components
explained_variance_ratio = pcas["SRV"].explained_variance_ratio_

# Get the feature weights (loadings) of the original features on the principal components
feature_weights = pcas["SRV"].components_.T

# Create a scatter trace for the cell data
trace_scatter = go.Scatter(
    x=pc1,
    y=pc2,
    mode="markers",
    marker=dict(
        size=5,  # Adjust marker size as per preference
        color="rgb(31, 119, 180)",  # Custom marker color
        opacity=0.8,
    ),
)

# Create arrow annotations for the feature weights
annotations = []
for i in range(len(feature_weights)):
    annotations.append(
        go.layout.Annotation(
            x=0,
            y=0,  # Starting point of the arrow
            xref="x",
            yref="y",
            ax=feature_weights[i, 0],
            ay=feature_weights[i, 1],  # Arrow direction and magnitude
            # text='Feature ' + str(i + 1),  # Label the arrow with feature number
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
        )
    )

# Create the layout for the biplot
layout = go.Layout(
    title="Tangent PCA Biplot",
    xaxis=dict(
        title="PC1 ({}% explained variance)".format(
            round(explained_variance_ratio[0] * 100, 2)
        )
    ),
    yaxis=dict(
        title="PC2 ({}% explained variance)".format(
            round(explained_variance_ratio[1] * 100, 2)
        )
    ),
    showlegend=False,
    annotations=annotations,
)

# Create the Figure object and add the scatter trace and annotations to it
fig = go.Figure(data=[trace_scatter], layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)
