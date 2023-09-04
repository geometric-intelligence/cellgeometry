import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves, ClosedDiscreteCurves
from geomstats.learning.frechet_mean import FrechetMean
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from plotly.subplots import make_subplots
from utils import experimental
import streamlit as st
import pandas as pd


st.sidebar.header("STEP 2: Compute Mean Shape")

st.write("# Compute Mean Shape")

if "selected_dataset" not in st.session_state:
    st.warning("👈 Please upload or Select Data from __Load Data__")
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


CLOSED_CURVES_SPACE = ClosedDiscreteCurves(Euclidean(dim=2))
CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2), k_sampling_points=n_sampling_points)


cells, cell_shapes = experimental.nolabel_preprocess(
    cells_list, len(cells_list), n_sampling_points
)


st.write(cell_shapes.shape)
st.session_state["cells"] = cells
st.session_state["cell_shapes"] = cell_shapes

if st.session_state["cell_lines"] is not None:
    cell_lines = st.session_state["cell_lines"]
    if st.session_state["treatment"] is not None:
        treatment = st.session_state["treatment"]
        exp_geo_traj = st.sidebar.toggle("Explore Geodesic Trajectory")


toggle_states = {}

if exp_geo_traj:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        selected_treatment = st.radio(
            "Select Treatment 👇", gs.unique(treatment), key="geodesic_treatment"
        )
    with col2:
        selected_cell_line1, selected_cell_line2 = st.multiselect(
            "Select Cell Line 👇",
            gs.unique(cell_lines),
            default=gs.unique(cell_lines)[:2],
            key="geodesic_cell_line",
        )

    def find_indices_with_selected_string(string_list, selected_string):
        indices = [
            index
            for index, string in enumerate(string_list)
            if string == selected_string
        ]
        return indices

    cell_indices_treatment = find_indices_with_selected_string(
        treatment, selected_treatment
    )
    cell_indices_cell_line1 = find_indices_with_selected_string(
        cell_lines, selected_cell_line1
    )
    cell_indices_cell_line2 = find_indices_with_selected_string(
        cell_lines, selected_cell_line2
    )

    cell_indices1 = list(set(cell_indices_treatment) & set(cell_indices_cell_line1))
    cell_indices2 = list(set(cell_indices_treatment) & set(cell_indices_cell_line2))

    geodesic_cell_index = st.select_slider(
        "Select a Cell Index", cell_indices1, key="geodesic_cell_index"
    )
    geodesic_cell_index2 = st.select_slider(
        "Select a Cell Index", cell_indices2, key="geodesic_cell_index2"
    )

    # st.write(pd.DataFrame(cell_indices))
    # st.write(pd.DataFrame(treatment))
    # st.write(pd.DataFrame(cell_lines))

    # i_start_rand = gs.random.randint(len(ds_proj["control"]["dunn"]))
    # i_end_rand = gs.random.randint(len(ds_proj["control"]["dlm8"]))

    cell_start = cell_shapes[geodesic_cell_index]
    cell_end = cell_shapes[geodesic_cell_index2]

    # print(i_start_rand, i_end_rand)

    geodesic_func = CURVES_SPACE_SRV.metric.geodesic(
        initial_point=cell_start, end_point=cell_end
    )

    n_times = 30
    times = gs.linspace(0.0, 1.0, n_times)
    geod_points = geodesic_func(times)

    # Create a subplots layout
    fig = make_subplots(
        rows=2,
        cols=n_times // 2,
    )

    # Add traces to the subplots
    for i, curve in enumerate(geod_points):
        row = i // (n_times // 2) + 1
        col = i % (n_times // 2) + 1
        trace = go.Scatter(x=curve[:, 0], y=curve[:, 1])
        fig.add_trace(trace, row=row, col=col)

    # Remove grid lines and axis lines
    for i in range(1, 3):  # for 2 rows
        for j in range(1, n_times // 2 + 1):  # for n_times//2 columns
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False,
                row=i,
                col=j,
            )
            fig.update_yaxes(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                visible=False,
                row=i,
                col=j,
            )

    # Update layout settings
    fig.update_layout(
        title="Geodesic between two cells",
        showlegend=False,
        xaxis_visible=False,
        yaxis_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)",
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig)

    # Create a new figure
    fig = go.Figure()

    # Loop through the geod_points and add a line plot for each time point
    for i in range(1, n_times - 1):
        fig.add_trace(
            go.Scatter(
                x=geod_points[i, :, 0],
                y=geod_points[i, :, 1],
                mode="lines+markers",
                marker=dict(size=8, color="goldenrod", symbol="circle"),
                line=dict(color="goldenrod", dash="dashdot"),
                hoverinfo="x+y",
                name=f"Time {i}",
            )
        )

    # Add the start cell (blue line plot)
    fig.add_trace(
        go.Scatter(
            x=geod_points[0, :, 0],
            y=geod_points[0, :, 1],
            mode="lines+markers",
            marker=dict(size=10, color="blue", symbol="circle"),
            line=dict(color="blue"),
            hoverinfo="x+y",
            name="Start Cell",
        )
    )

    # Add the end cell (red line plot)
    fig.add_trace(
        go.Scatter(
            x=geod_points[-1, :, 0],
            y=geod_points[-1, :, 1],
            mode="lines+markers",
            marker=dict(size=10, color="red", symbol="circle"),
            line=dict(color="red"),
            hoverinfo="x+y",
            name="End Cell",
        )
    )

    # Update the layout to add a title, gridlines, and axis labels
    fig.update_layout(
        title="Geodesic for the Square Root Velocity metric",
        xaxis=dict(title="X-axis Label", gridcolor="lightgray"),
        yaxis=dict(title="Y-axis Label", gridcolor="lightgray"),
        plot_bgcolor="white",
    )

    # Display the plot
    st.plotly_chart(fig)
# st.write(cell_shapes[0], cell_lines[0], treatment[0])


cell_shapes = gs.array(st.session_state["cell_shapes"])

compute_mean_shape = st.sidebar.toggle("Compute Mean Shape")

if compute_mean_shape:
    st.header("Explore Geodesic Trajectory Joining Two Cell Shapes")
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
        data=go.Scatter(
            x=x_coords, y=y_coords, mode="lines", line=dict(color=line_color)
        )
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
