import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.discrete_curves import DiscreteCurves, ClosedDiscreteCurves
from geomstats.learning.pca import TangentPCA
from geomstats.learning.frechet_mean import FrechetMean
from sklearn.decomposition import PCA
import pacmap
import plotly.express as px
import requests
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


st.sidebar.header("STEP 3: PACMAP")

st.write("# Dimension Reduction using PACMAP 👋")

st.markdown(
    """
PaCMAP (Pairwise Controlled Manifold Approximation) is a dimensionality reduction method that can be used for visualization, preserving both local and global structure of the data in original space.
"""
)

# st.help(pacmap.PaCMAP)


# # Send POST request
# response = requests.post(url, json=data)
# st.write("Sent request to server")
# # Check if the request was successful
# if response.status_code == 200:
#     transformed_data = response.json().get("transformed_data")
#     print("Transformed Data:", transformed_data)
# else:
#     print(f"Error {response.status_code}: {response.text}")


# if "cell_shapes" not in st.session_state:
#     st.warning(
#         "👈 Have you uploaded a zipped file of ROIs under Load Data? Afterwards, go the the Mean Shape page and run the analysis there."
#     )
#     st.stop()
# # cells = st.session_state["cells"]
# cell_shapes = st.session_state["cell_shapes"]


# if st.session_state["cell_lines"] is not None:
#     cell_lines = st.session_state["cell_lines"]
#     if st.session_state["treatment"] is not None:
#         treatment = st.session_state["treatment"]


# cells_flat = gs.reshape(cell_shapes, (len(cell_shapes), -1))


# st.write("Cells flat", cells_flat.shape)

# R1 = Euclidean(dim=1)
# CLOSED_CURVES_SPACE = ClosedDiscreteCurves(Euclidean(dim=2))
# CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2))
# mean = FrechetMean(CURVES_SPACE_SRV).fit(cell_shapes)

# n_components = st.slider("Select the Number of Sampling Points", 0, len(cells_flat), 10)


# st.write(treatment.shape)
# Perform PacMap dimensionality reduction

# st.help(pacmap.PaCMAP)

runPacmap = st.toggle("Run PACMAP Analysis", False)

if runPacmap:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### Number of Components")
        n_components = st.number_input(
            "Default = 3",
            min_value=3,
            max_value=None,
            key="n_components",
        )
        st.write("Input dimensions of the embedded space.", n_components)

    with col2:
        st.markdown("##### Number of Neighbors")
        neighbors_num = st.number_input(
            "Default = 10 ",
            min_value=1,
            max_value=None,
        )
        st.write(
            "Input number of neighbors considered for nearest neighbor pairs for local structure preservation."
        )

    with col3:
        st.markdown("##### Learning Rate")
        lr = st.number_input("Input learning rate. Default = 1.0")
        st.write("The learning rate is ")

    dist_options = st.selectbox(
        "Default = euclidean",
        ("euclidean", "manhattan", "angular", "hamming"),
    )

    st.write("Select distance metric.")
    model = pacmap.PaCMAP(n_components=st.session_state["n_components"])
    embedding = model.fit_transform([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    st.write(embedding.shape)
    # st.write(cell_lines.shape)

    # Visualize the embedding using Plotly Express
    # Create a scatter plot with coloring based on 'cell_lines' and symbols based on 'treatments'
    fig = px.scatter_3d(
        x=embedding[:, 0],
        y=embedding[:, 1],
        z=embedding[:, 2],
        color=gs.squeeze(cell_lines),  # differentiate by color based on cell_lines
        symbol=gs.squeeze(treatment),  # differentiate by symbol based on treatments
        title="PacMap Embedding",
        labels={"x": "Dimension 1", "y": "Dimension 2", "z": "Dimension 3"},
        color_discrete_sequence=px.colors.qualitative.Light24,  # use a color palette
    )

    # Update layout for better clarity, if needed
    fig.update_layout(legend_title_text="Cell Lines", legend_itemsizing="constant")

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    st.stop()
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(embedding[:, :3])
    pcd.estimate_normals()
    # Compute a mesh from the point cloud using the Ball-Pivoting Algorithm
    radii = [0.005, 0.01, 0.02, 0.04]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    # Compute normals for the point cloud
    # pcd.estimate_normals()

    # Access the vertices and faces
    vertices = np.asarray(mesh.vertices)
    # triangles = np.asarray(mesh.triangles)
    from scipy.spatial import Delaunay

    tri = Delaunay(vertices)
    triangles = tri.simplices
    st.write(
        """
### 3D Mesh Reconstruction from Reduced Dimension Embeddings

#### Description
The plot visualizes a 3D mesh that's been reconstructed from reduced dimension embeddings. Initially, a point cloud is formed from these embeddings. From this point cloud, a mesh is generated using the Ball-Pivoting Algorithm, which identifies possible triangles by virtually rolling a ball over the point cloud. The final visualization provides a geometric representation of the relationships and structures within the embeddings, showcasing patterns and clusters that might not be evident in a simple scatter plot.
             """
    )
    # fig = go.Figure(data=[go.Scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode='markers')])
    # Create a Mesh3D plot
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=triangles[:, 0],  # i, j, k define the vertices of the triangles
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.9,
            )
        ]
    )
    # Set plot layout
    # fig.update_layout(scene=dict(aspectmode='data'))

    # Show the plot
    st.plotly_chart(fig)

    st.markdown(
        """ ### Background on PACMAP

    PaCMAP optimizes the low dimensional embedding using three kinds of pairs of points: neighbor pairs (pair_neighbors), mid-near pair (pair_MN), and further pairs (pair_FP).

    Previous dimensionality reduction techniques focus on either local structure (e.g. t-SNE, LargeVis and UMAP) or global structure (e.g. TriMAP), but not both, although with carefully tuning the parameter in their algorithms that controls the balance between global and local structure, which mainly adjusts the number of considered neighbors. Instead of considering more neighbors to attract for preserving glocal structure, PaCMAP dynamically uses a special group of pairs -- mid-near pairs, to first capture global structure and then refine local structure, which both preserve global and local structure. For a thorough background and discussion on this work, please read [the paper](https://jmlr.org/papers/v22/20-1061.html).

    """
    )
# pcas = {}
st.header("Random Forest Classifier")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

selected_label = st.radio(
    "Select the labels to train the classifier", ("cell_lines", "treatment")
)
# Convert data to DataFrame
df = pd.DataFrame(embedding)
X = df.values  # Features (PCA components)
y = st.session_state[selected_label]  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy * 100:.2f}%")

# Extract feature importances
feature_importances = clf.feature_importances_
df_importance = pd.DataFrame(
    {
        "Features": list(range(n_components)),
        "Importance": feature_importances,
        # 'Treatment': y  # Assuming all treatments in the sample data are the same; adjust as needed
    }
)

# Create the plot
fig = px.bar(
    df_importance,
    x="Features",
    y="Importance",
    title="Feature Importances of PCA Components by Treatment",
    labels={"Importance": "Importance Value", "Features": "PCA Components"},
)

st.plotly_chart(fig)

# st.write(mean.estimate_)
# logs = CURVES_SPACE_SRV.metric.log(cells_flat, base_point=mean.estimate_)
# logs = gs.reshape(logs, (len(cells), -1))
# PCA(n_components).fit(logs)


# CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2))
# mean = FrechetMean(CURVES_SPACE_SRV).fit(cell_shapes)
# logs = []
# for one_cell in cell_shapes:
#     one_log = CURVES_SPACE_SRV.metric.log(one_cell, base_point=mean.estimate_)
#     logs.append(one_log)
# logs = gs.array(logs)  # same shape as cell_shapes here
# logs = gs.reshape(logs, (len(cells), -1))
# PCA(n_components).fit(logs)

# for one_shape in cells_flat:
#     one_log = CURVES_SPACE_SRV.metric.log(one_shape, base_point=mean.estimate_)

# for one_shape in cell_shapes:
#     one_log = CURVES_SPACE_SRV.metric.log(one_shape, base_point=mean.estimate_)
# CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2))

# pcas["SRV"] = TangentPCA(n_components=n_components, space=CURVES_SPACE_SRV).fit(cell_shapes)

# CURVES_SPACE_SRV = DiscreteCurves(Euclidean(dim=2))
# mean = FrechetMean(CURVES_SPACE_SRV).fit(cell_shapes)
# pcas["SRV"] = TangentPCA(n_components=n_components, space=CURVES_SPACE_SRV).fit(cell_shapes, base_point=mean.estimate_)
# pcas = {}
# pcas["Linear"] = PCA(n_components=n_components).fit(cells_flat)
# pcas["Linear_components"] = PCA(n_components=n_components).fit_transform(cells_flat)
# pcas["SRV"] = TangentPCA(n_components=n_components, space=mean).fit(cell_shapes)
# pcas["SRV_components"] = TangentPCA(
#     n_components=n_components, space=mean
# ).fit_transform(cell_shapes)

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

st.stop()

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

st.markdown(
    """

### Tangent PCA Analysis

Linear dimensionality reduction using
    Singular Value Decomposition of the
    Riemannian Log of the data at the tangent space
    of the Frechet mean.
    """
)

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
        title=f"PC1 ({round(explained_variance_ratio[0] * 100, 2)}% explained variance)"
    ),
    yaxis=dict(
        title=f"PC2 ({round(explained_variance_ratio[1] * 100, 2)}% explained variance)"
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
        title=f"PC1 ({round(explained_variance_ratio[0] * 100, 2)}% explained variance)"
    ),
    yaxis=dict(
        title=f"PC2 ({round(explained_variance_ratio[1] * 100, 2)}% explained variance)"
    ),
    showlegend=False,
    annotations=annotations,
)

# Create the Figure object and add the scatter trace and annotations to it
fig = go.Figure(data=[trace_scatter], layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)
