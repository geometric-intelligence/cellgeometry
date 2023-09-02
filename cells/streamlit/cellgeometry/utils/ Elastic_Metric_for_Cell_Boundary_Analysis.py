import sys
import geomstats.backend as gs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import R2, DiscreteCurves, ClosedDiscreteCurves

from geomstats.learning.frechet_mean import FrechetMean

# import utils
from utils import experimental as experimental
from utils import basic as basic
import streamlit as st


sys.path.append("/app/utils")


st.set_page_config(
    page_title="Elastic Metric for Cell Boundary Analysis", page_icon="📈"
)

st.markdown("# Shape Analysis of Cancer Cells")
st.sidebar.header("Shape Analysis of Cancer Cells")
st.write(
    """This notebook studies Osteosarcoma (bone cancer) cells and the impact of drug treatment on their morphological shapes, by analyzing cell images obtained from fluorescence microscopy.

This analysis relies on the elastic metric between discrete curves from Geomstats. We will study to which extent this metric can detect how the cell shape is associated with the response to treatment."""
)

dataset_name = "osteosarcoma"

n_sampling_points = st.slider("Select the Number of Sampling Points", 0, 250, 100)
n_cells = 650
# n_sampling_points = 100
labels_a_name = "lines"
labels_b_name = "treatments"

quotient = ["rotation"]  # ["scaling"] #, "rotation"]
do_not_quotient = False


if dataset_name == "osteosarcoma":
    (
        cells,
        cell_shapes,
        labels_a,
        labels_b,
    ) = experimental.load_treated_osteosarcoma_cells(
        n_cells=n_cells, n_sampling_points=n_sampling_points, quotient=quotient
    )
else:
    pass


labels_a_dict = {lab: i_lab for i_lab, lab in enumerate(np.unique(labels_a))}
labels_b_dict = {lab: i_lab for i_lab, lab in enumerate(np.unique(labels_b))}

print(f'Dictionary associated to label "{labels_a_name}":')
print(labels_a_dict)
print(f'Dictionary associated to label "{labels_b_name}":')
print(labels_b_dict)

if do_not_quotient:
    cell_shapes = cells

n_cells_to_plot = 10

fig = plt.figure(figsize=(16, 6))
count = 1
for label_b in np.unique(labels_b):
    for i_lab_a, label_a in enumerate(np.unique(labels_a)):
        cell_data = [
            cell
            for cell, lab_a, lab_b in zip(cell_shapes, labels_a, labels_b)
            if lab_a == label_a and lab_b == label_b
        ]
        for i_to_plot in range(n_cells_to_plot):
            cell = gs.random.choice(a=cell_data)
            fig.add_subplot(
                len(np.unique(labels_b)),
                len(np.unique(labels_a)) * n_cells_to_plot,
                count,
            )
            count += 1
            plt.plot(cell[:, 0], cell[:, 1], color=f"C{i_lab_a}")
            plt.axis("equal")
            plt.axis("off")
            if i_to_plot == n_cells_to_plot // 2:
                plt.title(f"{label_a}   -   {label_b}", fontsize=20)
st.pyplot(fig)

# Define shape space
R1 = Euclidean(dim=1)
CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)
CURVES_SPACE = DiscreteCurves(R2)
SRV_METRIC = CURVES_SPACE.srv_metric
L2_METRIC = CURVES_SPACE.l2_curves_metric

ELASTIC_METRIC = {}
AS = [1, 2, 0.75, 0.5, 0.25, 0.01]  # , 1.6] #, 1.4, 1.2, 1, 0.5, 0.2, 0.1]
BS = [0.5, 1, 0.5, 0.5, 0.5, 0.5]  # , 2, 2, 2, 2, 2, 2, 2]
for a, b in zip(AS, BS):
    ELASTIC_METRIC[a, b] = DiscreteCurves(R2, a=a, b=b).elastic_metric
METRICS = {}
METRICS["Linear"] = L2_METRIC
METRICS["SRV"] = SRV_METRIC

means = {}

means["Linear"] = gs.mean(cell_shapes, axis=0)
means["SRV"] = (
    FrechetMean(metric=SRV_METRIC, method="default").fit(cell_shapes).estimate_
)

for a, b in zip(AS, BS):
    means[a, b] = (
        FrechetMean(metric=ELASTIC_METRIC[a, b], method="default")
        .fit(cell_shapes)
        .estimate_
    )

st.header("Sample Means")
st.markdown(
    "We compare results when computing the mean cell versus the mean cell shapes with different elastic metrics."
)
fig = plt.figure(figsize=(18, 8))

ncols = len(means) // 2

for i, (mean_name, mean) in enumerate(means.items()):
    ax = fig.add_subplot(2, ncols, i + 1)
    ax.plot(mean[:, 0], mean[:, 1], "black")
    ax.set_aspect("equal")
    ax.axis("off")
    axs_title = mean_name
    if mean_name not in ["Linear", "SRV"]:
        a = mean_name[0]
        b = mean_name[1]
        ratio = a / (2 * b)
        mean_name = f"Elastic {mean_name}\n a / (2b) = {ratio}"
    ax.set_title(mean_name)

st.pyplot(fig)


fig = plt.figure(figsize=(18, 8))

ncols = len(means) // 2

for i, (mean_name, mean) in enumerate(means.items()):
    ax = fig.add_subplot(2, ncols, i + 1)
    mean = CLOSED_CURVES_SPACE.projection(mean)
    ax.plot(mean[:, 0], mean[:, 1], "black")
    ax.set_aspect("equal")
    ax.axis("off")
    axs_title = mean_name
    if mean_name not in ["Linear", "SRV"]:
        a = mean_name[0]
        b = mean_name[1]
        ratio = a / (2 * b)
        mean_name = f"Elastic {mean_name}\n a / (2b) = {ratio}"
    ax.set_title(mean_name)


st.markdown(
    "__Remark:__ Unfortunately, there are some numerical issues with the projection in the space of closed curves, as shown by the V-shaped results above."
)

st.markdown(
    "Since ratios of 1 give the same results as for the SRV metric, we only select AS, BS with a ratio that is not 1 for the elastic metrics."
)

st.markdown(
    "We also continue the analysis with the space of open curves, as opposed to the space of closed curves, for the numerical issues observed above."
)


NEW_AS = [0.75, 0.5, 0.25, 0.01]  # , 1.6] #, 1.4, 1.2, 1, 0.5, 0.2, 0.1]
NEW_BS = [0.5, 0.5, 0.5, 0.5]  # , 2, 2, 2, 2, 2, 2, 2]

st.markdown("## Distances to the Mean")

# We multiply the distances by a 100, for visualization purposes. It amounts to a change of units.
dists = {}

dists["Linear"] = [
    100 * gs.linalg.norm(means["Linear"] - cell) / n_sampling_points
    for cell in cell_shapes
]

dists["SRV"] = [
    100 * SRV_METRIC.dist(means["SRV"], cell) / n_sampling_points
    for cell in cell_shapes
]

for a, b in zip(NEW_AS, NEW_BS):
    dists[a, b] = [
        100 * ELASTIC_METRIC[a, b].dist(means[a, b], cell) / n_sampling_points
        for cell in cell_shapes
    ]


dists_summary = pd.DataFrame(
    data={
        labels_a_name: labels_a,
        labels_b_name: labels_b,
        "Linear": dists["Linear"],
        "SRV": dists["SRV"],
    }
)

for a, b in zip(NEW_AS, NEW_BS):
    dists_summary[f"Elastic({a}, {b})"] = dists[a, b]

st.dataframe(dists_summary)
# SAVEFIG = True
# if SAVEFIG:
#     figs_dir = os.path.join(work_dir, f"cells/saved_figs/{dataset_name}")
#     if not os.path.exists(figs_dir):
#         os.makedirs(figs_dir)
#     print(f"Will save figs to {figs_dir}")
#     from datetime import datetime

#     now = datetime.now().strftime("%Y%m%d_%H_%M_%S")
#     print(now)
