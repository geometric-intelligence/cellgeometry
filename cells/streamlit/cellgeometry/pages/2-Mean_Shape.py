import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves, ClosedDiscreteCurves
from geomstats.learning.frechet_mean import FrechetMean
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from utils import experimental
import streamlit as st


st.sidebar.header("STEP 2: Compute Mean Shape")

st.write("# Compute Mean Shape")

if not st.session_state["selected_dataset"]:
    st.warning("👈 Please upload a zipped file of ROIs under Load Data")
    st.stop()

# Display the uploaded data
st.info(f"Uploaded data: {st.session_state['selected_dataset']}")

st.markdown(
    """
    ## Step Zero

    👈 If you have not already uploaded your data, please select the __Load Data__ page and follow the instructions. The format is important, so please read carefully.

    ## Analyzing Cell Data

    Now we will start analyzing our data. The first step is preprocessing our data, specifically interpolating, removing duplicates, and quotienting.
"""
)


upload_folder = st.session_state["selected_dataset"]
cells_list = st.session_state["cells_list"]
# st.session_state["cell_shapes"] = None
# st.write(st.session_state["cell_shapes"])
n_sampling_points = st.slider("Select the Number of Sampling Points", 0, 250, 150)
st.session_state["n_sampling_points"] = n_sampling_points

if "cell_shapes" not in st.session_state:
    cells, cell_shapes = experimental.nolabel_preprocess(
        cells_list, len(cells_list), n_sampling_points
    )
    st.write(cell_shapes.shape)
    st.session_state["cells"] = cells
    st.session_state["cell_shapes"] = cell_shapes

if st.session_state["cell_lines"] is not None:
    cell_lines = st.session_state["cell_lines"]
    exp_geo_traj = st.sidebar.checkbox("Explore Geodesic Trajectory")

if st.session_state["treatment"] is not None:
    treatment = st.session_state["treatment"]

# st.write(cell_shapes[0], cell_lines[0], treatment[0])

CLOSED_CURVES_SPACE = ClosedDiscreteCurves(Euclidean(dim=2))
CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2), k_sampling_points=n_sampling_points)


if exp_geo_traj:
    st.header("Explore Geodesic Trajectory Joining Two Cell Shapes")

    # i_start_rand = gs.random.randint(len(ds_proj["control"]["dunn"]))
    # i_end_rand = gs.random.randint(len(ds_proj["control"]["dlm8"]))

    # cell_start = ds_align["control"]["dunn"][i_start_rand]
    # cell_end = ds_align["control"]["dlm8"][i_end_rand]

    # print(i_start_rand, i_end_rand)

    # geodesic_func = CURVES_SPACE_SRV.metric.geodesic(initial_point=cell_start, end_point=cell_end)

    # n_times = 30
    # times = gs.linspace(0.0, 1.0, n_times)
    # geod_points = geodesic_func(times)


# SRV_METRIC = CURVES_SPACE.srv_metric
# L2_METRIC = CURVES_SPACE.l2_curves_metric

# ELASTIC_METRIC = {}

# input_string_AS = st.text_input("Elastic Metric AS (Use comma-seperated values)", "1")

# # Convert string to list of integers
# AS = [float(num) for num in input_string_AS.split(",")]

# input_string_BS = st.text_input("Elastic Metric BS (Use comma-seperated values)", "0.5")
# BS = [float(num) for num in input_string_BS.split(",")]
# # AS = [1, 2, 0.75, 0.5, 0.25, 0.01] #, 1.6] #, 1.4, 1.2, 1, 0.5, 0.2, 0.1]
# # BS = [0.5, 1, 0.5, 0.5, 0.5, 0.5] #, 2, 2, 2, 2, 2, 2, 2]


# for a, b in zip(AS, BS):
#     ELASTIC_METRIC[a, b] = CURVES_SPACE.metric.geodesic(
#         initial_point=a, end_point=b
#     )  # .elastic_metric
# METRICS = {}
# METRICS["Linear"] = L2_METRIC
# METRICS["SRV"] = SRV_METRIC


# means = {}

# means["Linear"] = gs.mean(cell_shapes, axis=0)
# means["SRV"] = (
#     FrechetMean(metric=SRV_METRIC, method="default").fit(cell_shapes).estimate_
# )


# for a, b in zip(AS, BS):
#     means[a, b] = (
#         FrechetMean(metric=ELASTIC_METRIC[a, b], method="default")
#         .fit(cell_shapes)
#         .estimate_
#     )


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

# ncols = len(means) // 2

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
# st.write(cell_shapes)
cell_shapes = gs.array(cell_shapes)


means = FrechetMean(CURVES_SPACE_SRV)
# st.write(means)
means.fit(cell_shapes[:500])

mean_estimate = means.estimate_

# plt.plot(mean_estimate[:, 0], mean_estimate[:, 1], "black")

# Extract x and y coordinates
x_coords = mean_estimate[:, 0]
y_coords = mean_estimate[:, 1]

line_color = "rgb(255, 0, 191)"

# Create a Plotly scatter plot
fig = go.Figure(
    data=go.Scatter(x=x_coords, y=y_coords, mode="lines", line=dict(color=line_color))
)

# Customize layout
fig.update_layout(
    title="Mean Estimate",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)


mean_estimate_clean = mean_estimate[~gs.isnan(gs.sum(mean_estimate, axis=1)), :]
mean_estimate_aligned = 1.55 * (
    mean_estimate_clean - gs.mean(mean_estimate_clean, axis=0)
)


# Create a Plotly figure
fig = go.Figure()

# Plot cell shapes
for cell in cell_shapes:
    fig.add_trace(
        go.Scatter(
            x=cell[:, 0],
            y=cell[:, 1],
            mode="lines",
            line=dict(color="lightgrey", width=1),
        )
    )

# Plot mean estimate
fig.add_trace(
    go.Scatter(
        x=mean_estimate_aligned[:, 0],
        y=mean_estimate_aligned[:, 1],
        mode="lines",
        line=dict(color="black", width=2),
        name="Mean cell",
    )
)

# Customize layout
fig.update_layout(
    title="Cell Shapes and Mean Estimate",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    legend=dict(font=dict(size=12)),
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)


mean_estimate_aligned_bis = gs.vstack(
    [mean_estimate_aligned[4:], mean_estimate_aligned[-1]]
)


cells_to_plot = cell_shapes[gs.random.randint(len(cell_shapes), size=300)]
points_to_plot = cells_to_plot.reshape(-1, 2)

z = gaussian_kde(points_to_plot.T)(points_to_plot.T)
z_norm = z / z.max()

# Create a Plotly figure
fig = go.Figure()

# Scatter plot for points
fig.add_trace(
    go.Scatter(
        x=points_to_plot[:, 0],
        y=points_to_plot[:, 1],
        mode="markers",
        marker=dict(color=z_norm, size=10, opacity=0.2),
    )
)

# Plot mean estimate
fig.add_trace(
    go.Scatter(
        x=mean_estimate_aligned_bis[:, 0],
        y=mean_estimate_aligned_bis[:, 1],
        mode="lines",
        line=dict(color="black", width=2),
        name="Mean cell",
    )
)

# Customize layout
fig.update_layout(
    title="Global mean shape superimposed on the dataset of cells",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    legend=dict(font=dict(size=12)),
    title_font_size=14,
)

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

# import numpy as np


# # Create a density heatmap instead of scatter points
# hist_data = np.histogram2d(points_to_plot[:, 0], points_to_plot[:, 1], bins=(100, 100))
# heatmap = go.Heatmap(z=hist_data[0], x=hist_data[1], y=hist_data[2], colorscale='Viridis', opacity=0.6)

# # Plot mean estimate with gradient color
# mean_shape_line = go.Scatter(
#     x=mean_estimate_aligned_bis[:, 0], y=mean_estimate_aligned_bis[:, 1],
#     mode='lines',
#     line=dict(width=2, color='mediumpurple'),
#     name='Mean cell',
#     hovertext=[f"X: {x}, Y: {y}" for x, y in zip(mean_estimate_aligned_bis[:, 0], mean_estimate_aligned_bis[:, 1])]
# )

# # Customize layout with a unique theme
# layout = go.Layout(
#     title="Global mean shape superimposed on the density heatmap of cells",
#     xaxis_title="X-axis",
#     yaxis_title="Y-axis",
#     legend=dict(font=dict(size=12)),
#     title_font_size=14,
#     # template="plotly_dark",  # Use a dark theme for a unique look
# )

# fig = go.Figure(data=[heatmap, mean_shape_line], layout=layout)

# # Display the Plotly figure in Streamlit
# st.plotly_chart(fig)


#############################################################
# ncols = 2  # Define ncols here, the number of columns in your subplot grid
# nrows = (
#     int(len(means) / ncols) + len(means) % ncols
# )  # calculate number of rows based on length of means

# # Create subplots with defined title font and size
# fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(map(str, means.keys())))

# # Define a color palette
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# for i, (mean_name, mean) in enumerate(means.items()):
#     mean = CLOSED_CURVES_SPACE.projection(mean)
#     row = i // ncols + 1
#     col = i % ncols + 1
#     color = colors[i % len(colors)]  # Select color from palette
#     fig.add_trace(
#         go.Scatter(
#             x=mean[:, 0],
#             y=mean[:, 1],
#             mode="lines",
#             name=str(mean_name),
#             line=dict(color=color, width=2),
#         ),
#         row=row,
#         col=col,
#     )

#     if mean_name not in ["Linear", "SRV"]:
#         a = mean_name[0]
#         b = mean_name[1]
#         ratio = a / (2 * b)
#         mean_name_str = f"Elastic {mean_name} | a/(2b) = {ratio}"
#         fig.layout.annotations[i]["text"] = str(mean_name_str)  # update subplot title

# fig.update_layout(
#     showlegend=False,
#     plot_bgcolor="#fafafa",
#     paper_bgcolor="#fafafa",
#     font=dict(family="Courier New, monospace", size=12, color="black"),
# )

# # Update xaxis and yaxis parameters to be invisible
# for i in range(1, nrows * ncols + 1):
#     fig.update_xaxes(
#         showticklabels=False,
#         showgrid=False,
#         zeroline=False,
#         visible=False,
#         row=(i - 1) // ncols + 1,
#         col=(i - 1) % ncols + 1,
#     )
#     fig.update_yaxes(
#         showticklabels=False,
#         showgrid=False,
#         zeroline=False,
#         visible=False,
#         row=(i - 1) // ncols + 1,
#         col=(i - 1) % ncols + 1,
#     )


# # Display the plot
# # fig.show()
# st.plotly_chart(fig)
#############################################################
