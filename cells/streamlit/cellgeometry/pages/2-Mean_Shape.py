import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves, ClosedDiscreteCurves
from geomstats.learning.frechet_mean import FrechetMean
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import experimental
import streamlit as st


st.sidebar.header("STEP 2: Compute Mean Shape")

st.write("# Compute Mean Shape")

# Display the uploaded data
st.info(f"Uploaded data: {st.session_state['upload_folder']}")

st.markdown(
    """
    ## Step Zero

    👈 If you have not already uploaded your data, please select the __Load Data__ page and follow the instructions. The format is important, so please read carefully.

    ## Analyzing Cell Data

    Now we will start analyzing our data. The first step is preprocessing our data, specifically interpolating, removing duplicates, and quotienting.
"""
)

if "upload_folder" not in st.session_state:
    st.warning("👈 Please upload a zipped file of ROIs under Load Data")
    st.stop()

upload_folder = st.session_state["upload_folder"]
cells_list = st.session_state["cells_list"]

n_sampling_points = st.slider("Select the Number of Sampling Points", 0, 100, 50)
st.session_state["n_sampling_points"] = n_sampling_points
cells, cell_shapes = experimental.nolabel_preprocess(
    cells_list, len(cells_list), n_sampling_points
)

st.session_state["cells"] = cells
st.session_state["cell_shapes"] = cell_shapes

R1 = Euclidean(dim=1)
CLOSED_CURVES_SPACE = ClosedDiscreteCurves(Euclidean(dim=2))
CURVES_SPACE = DiscreteCurves(Euclidean(dim=2), k_sampling_points=n_sampling_points)
# SRV_METRIC = CURVES_SPACE.srv_metric
# L2_METRIC = CURVES_SPACE.l2_curves_metric

ELASTIC_METRIC = {}

input_string_AS = st.text_input("Elastic Metric AS (Use comma-seperated values)", "1")

# Convert string to list of integers
AS = [float(num) for num in input_string_AS.split(",")]

input_string_BS = st.text_input("Elastic Metric BS (Use comma-seperated values)", "0.5")
BS = [float(num) for num in input_string_BS.split(",")]
# AS = [1, 2, 0.75, 0.5, 0.25, 0.01] #, 1.6] #, 1.4, 1.2, 1, 0.5, 0.2, 0.1]
# BS = [0.5, 1, 0.5, 0.5, 0.5, 0.5] #, 2, 2, 2, 2, 2, 2, 2]


for a, b in zip(AS, BS):
    ELASTIC_METRIC[a, b] = CURVES_SPACE.metric.geodesic(
        initial_point=a, end_point=b
    )  # .elastic_metric
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


# fig = plt.figure(figsize=(18, 8))

# ncols = len(means) // 2

# for i, (mean_name, mean) in enumerate(means.items()):
#     ax = fig.add_subplot(2, ncols, i+1)
#     ax.plot(mean[:, 0], mean[:, 1], "black")
#     ax.set_aspect("equal")
#     ax.axis("off")
#     axs_title = mean_name
#     if mean_name not in ["Linear", "SRV"]:
#         a = mean_name[0]
#         b = mean_name[1]
#         ratio = a / (2 * b)
#         mean_name = f"Elastic {mean_name}\n a / (2b) = {ratio}"
#     ax.set_title(mean_name)

# st.pyplot(fig)


# fig = plt.figure(figsize=(18, 8))

ncols = len(means) // 2

# for i, (mean_name, mean) in enumerate(means.items()):
#     ax = fig.add_subplot(2, ncols, i+1)
#     mean = CLOSED_CURVES_SPACE.projection(mean)
#     ax.plot(mean[:, 0], mean[:, 1], "black")
#     ax.set_aspect("equal")
#     ax.axis("off")
#     axs_title = mean_name
#     if mean_name not in ["Linear", "SRV"]:
#         a = mean_name[0]
#         b = mean_name[1]
#         ratio = a / (2 * b)
#         mean_name = f"Elastic {mean_name}\n a / (2b) = {ratio}"
#     ax.set_title(mean_name)

# st.pyplot(fig)


ncols = 2  # Define ncols here, the number of columns in your subplot grid
nrows = (
    int(len(means) / ncols) + len(means) % ncols
)  # calculate number of rows based on length of means

# Create subplots with defined title font and size
fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(map(str, means.keys())))

# Define a color palette
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, (mean_name, mean) in enumerate(means.items()):
    mean = CLOSED_CURVES_SPACE.projection(mean)
    row = i // ncols + 1
    col = i % ncols + 1
    color = colors[i % len(colors)]  # Select color from palette
    fig.add_trace(
        go.Scatter(
            x=mean[:, 0],
            y=mean[:, 1],
            mode="lines",
            name=str(mean_name),
            line=dict(color=color, width=2),
        ),
        row=row,
        col=col,
    )

    if mean_name not in ["Linear", "SRV"]:
        a = mean_name[0]
        b = mean_name[1]
        ratio = a / (2 * b)
        mean_name_str = f"Elastic {mean_name} | a/(2b) = {ratio}"
        fig.layout.annotations[i]["text"] = str(mean_name_str)  # update subplot title

fig.update_layout(
    showlegend=False,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#fafafa",
    font=dict(family="Courier New, monospace", size=12, color="black"),
)

# Update xaxis and yaxis parameters to be invisible
for i in range(1, nrows * ncols + 1):
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        row=(i - 1) // ncols + 1,
        col=(i - 1) % ncols + 1,
    )
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        row=(i - 1) // ncols + 1,
        col=(i - 1) % ncols + 1,
    )


# Display the plot
# fig.show()
st.plotly_chart(fig)
