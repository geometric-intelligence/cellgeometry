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

from utils import experimental


st.write("# Principal Component Analysis (PCA) 👋")

cells = st.session_state["cells"] 
cell_shapes = st.session_state["cell_shapes"] 
if cells.size == 0:
  st.warning('👈 Have you uploaded a zipped file of ROIs under Load Data? Afterwards, go the the Cell Shear page and run the analysis there.')
  st.stop()


cells_flat = gs.reshape(cells, (len(cells), -1))

R1 = Euclidean(dim=1)
CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)
CURVES_SPACE = DiscreteCurves(R2)
SRV_METRIC = CURVES_SPACE.srv_metric
L2_METRIC = CURVES_SPACE.l2_curves_metric

n_components = st.slider('Select the Number of Components', 0, len(cells_flat), 10)

pcas = {}
pcas["Linear"] = PCA(n_components=n_components).fit(cells_flat)
pcas["SRV"] = TangentPCA(n_components=n_components, metric=SRV_METRIC).fit(cell_shapes)

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


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot with 1 row and 2 columns
fig = make_subplots(rows=1, cols=2, shared_yaxes=True)

for i, metric_name in enumerate(["Linear", "SRV"]):
    tangent = "Tangent " if metric_name == "SRV" else ""
    first_pc_explains = 100*sum(pcas[metric_name].explained_variance_ratio_[:1])
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pcas[metric_name].explained_variance_ratio_))), 
            y=pcas[metric_name].explained_variance_ratio_, 
            mode='lines',
            name=f"{metric_name}",
            legendgroup=f"{metric_name}", # Legend grouping
            hovertemplate="Number of PCS: %{x}<br>Explained Variance: %{y:.3f}", # Hover text
        ),
        row=1, 
        col=i+1
    )
    
    fig.update_xaxes(title_text="Number of PCS", row=1, col=i+1)
    fig.update_yaxes(title_text="Explained variance", row=1, col=i+1)

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
        legend=dict(
            y=0.5,
            traceorder="reversed",
            font=dict(size=10),
            yanchor="middle"
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        paper_bgcolor="#fafafa",
    )

# fig.show()
st.plotly_chart(fig)
    
# Explained Variances 
lin_explained = sum(pcas["Linear"].explained_variance_ratio_[:2])
srv_explained = sum(pcas["SRV"].explained_variance_ratio_[:2])

st.write("## Discussion")

st.write(f"The first two components of the PCA with the __Linear metric__ explains {lin_explained:.2f} of the variance")

st.write(f"The first two components of the __Tangent PCA__ with the SRV metric explains {srv_explained:.2f} of the variance.")