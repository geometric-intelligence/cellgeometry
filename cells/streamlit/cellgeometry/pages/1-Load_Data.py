import os
import time
import sys
from utils.data_utils import (
    build_rois,
    find_all_instances,
    get_files_from_folder,
    check_file_extensions,
    parse_coordinates,
    get_file_or_folder_type,
    get_csv_txt_files,
)
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import glob


st.sidebar.header("STEP 1: Load Data")
sys.path.append("/app/utils")

# How we get the current user after successful login
username = st.session_state["username"]

#########################################
# REMOVE BRUH
if username is None:
    username = "bruh"
    st.session_state["username"] = username
#########################################

current_time = time.localtime()
year = time.strftime("%Y", current_time)
day_of_year = time.strftime("%j", current_time)
time_string = time.strftime("%H%M%S", current_time)
upload_folder = f"/app/data/{username}"

if "current_time_string" not in st.session_state:
    current_time_string = f"{year}{day_of_year}-{time_string}"
    st.session_state["current_time_string"] = current_time_string


if "cells_list" not in st.session_state:
    st.session_state["cells_list"] = True

if "config_option" not in st.session_state:
    st.session_state["config_option"] = "Upload a File"

st.write("# Load Your Cell Data 👋")

st.divider()
st.markdown(
    """
## Getting Started

We currently support an ROI zip folder created by FIJI/ImageJ.
What this means is you may have a folder structure as follows:
```
    └── Cropped_Images
        ├── Bottom_plank_0
        │   ├── Averaged_ROI
        │   ├── Data
        │   ├── Data_Filtered
        │   ├── Labels
        │   ├── OG
        │   ├── Outlines
        │   └── ROIs  <---- Folder of zipped ROIs
```
You can simply upload this ROIs folder and we will load your data for you.
We plan on supporting data given in `xy` coordinate format from `JSON` and CSV/TXT files.
Your chosen data structure __must__ contain `x` and `y` for the program to correctly parse
and load your data.
"""
)
st.divider()

st.header("Step 1. Select Input Data")


def build_and_load_data(upload_folder):

    if "Folder" in get_file_or_folder_type(upload_folder):
        # Get the list of files in the upload folder
        files = get_files_from_folder(upload_folder)
    else:
        files = [upload_folder]
    # Check the file extensions
    extension = check_file_extensions(files)

    if extension[0] in [".csv", ".txt"]:

        # Build a dictionary of all the ROIs
        dict_rois = parse_coordinates(files[0])

        # Extract the cells into a list
        cells_list = list(dict_rois.values())

        st.session_state["cells_list"] = cells_list

        st.success(f"Successfully Loaded {len(cells_list)} cells.", icon="✅")

    else:
        # Build a dictionary of all the ROIs
        dict_rois = build_rois(upload_folder)

        # Extract the cells
        cells_list = []
        find_all_instances(dict_rois, "x", "y", cells_list)
        st.session_state["cells_list"] = cells_list

        st.success(f"Successfully Loaded {len(cells_list)} cells.", icon="✅")


# Check if the upload folder exists, and create it if it doesn't
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    st.session_state["upload_folder"] = upload_folder
    st.info(f"Upload folder created: {upload_folder}")


config_option = st.radio(
    "How would you like to provide the data?",
    ["Upload a File", "Choose an Uploaded File"],
    captions=[
        "Data must be in `.zip` for ImageJ ROI, `.txt` accepted",
        "Select previously uploaded data.",
    ],
    key="config_option",
)


def check_file_extension(file):
    return os.path.splitext(file.name)[1]


def handle_uploaded_file(uploaded_file, destination_folder):
    # Ensure destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    file_path = os.path.join(destination_folder, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


if st.session_state["config_option"] == "Upload a File":
    uploaded_files = st.file_uploader(
        "Upload ___Cell Data___ in one or multiple files (zip, csv, txt)",
        type=["zip", "csv", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                extension = check_file_extension(uploaded_file)

                # If file is a ZIP, create a new sub-folder
                if extension == ".zip":
                    new_upload_folder = os.path.join(
                        upload_folder, "ROIs-" + st.session_state["current_time_string"]
                    )
                    st.session_state["selected_dataset"] = new_upload_folder
                    handle_uploaded_file(uploaded_file, new_upload_folder)
                else:
                    handle_uploaded_file(uploaded_file, upload_folder)

            build_and_load_data(st.session_state["selected_dataset"])


elif st.session_state["config_option"] == "Choose an Uploaded File":

    if upload_folder and os.path.exists(upload_folder):

        # List all files and folders in the directory
        files = [
            f
            for f in os.listdir(upload_folder)
            if os.path.exists(os.path.join(upload_folder, f))
        ]

        if not files:
            st.warning("No files found in the directory!")
            # st.stop()

        # Display the files in a dropdown
        selected_file = st.selectbox("Choose a file:", files)
        st.session_state["selected_dataset"] = os.path.join(
            upload_folder, selected_file
        )
        selection_type = get_file_or_folder_type(st.session_state["selected_dataset"])
        build_and_load_data(st.session_state["selected_dataset"])
        st.info(f"{selection_type} Selected: {selected_file}")

        # if "Folder" in selection_type:

        #     selected_folder_files = [f for f in os.listdir(st.session_state["selected_dataset"]) if os.path.isfile(os.path.join(st.session_state["selected_dataset"], f))]
        #     view_selected_folder_files = st.selectbox("Choose a file:", selected_folder_files)

        # files = [
        #     f
        #     for f in os.listdir(upload_folder)
        #     if os.path.isfile(os.path.join(upload_folder, f))
        # ]
        # st.write(files)
        # if files:
        #     selected_file = st.selectbox("Select a previously uploaded file:", files)
        #     st.session_state["selected_dataset"] = selected_file
        #     st.info(f"You selected {selected_file}")
        #     build_and_load_data(upload_folder)
        #     # previewVisualization(st.session_state["cells_list"])
        # else:
        #     st.warning("No files found in the directory!")
    else:
        st.warning("No files have been uploaded yet.")


st.header("Step 2. Select Labels (Optional)")

st.markdown(
    """
            Your labels should be in a `.csv` or `.txt` file. Within this file, each row corresponds to the cell number and that row should contain the label for that cell. For example, if you have 10 cells, your file should have 10 rows. The first row should be the label for cell 0, the second row should be the label for cell 1, and so on. The label can be any string, but it should be unique for each cell.

            For example, if you are uploading the treatments for each cell you can use `control`, `cytd`, `jasp`, etc.
            """
)

hasTreatments = st.checkbox("Treatments")

treatment_col1, treatment_col2 = st.columns([0.8, 0.2], gap="medium")
with treatment_col1:
    if hasTreatments:
        uploaded_labels = st.file_uploader(
            "Upload ___Treatments___ in one or multiple files (`csv`, `txt`)",
            type=["csv", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_labels:
            with st.spinner("Processing files..."):
                for uploaded_label in uploaded_labels:

                    handle_uploaded_file(uploaded_label, upload_folder)

                st.session_state["treatment"] = pd.read_csv(
                    uploaded_label, header=None
                ).values

        if st.session_state["config_option"] == "Choose an Uploaded File":

            # Display the files in a dropdown
            selected_treatment_file = st.selectbox(
                "Choose a file:",
                get_csv_txt_files(upload_folder),
                key="selected_treatment_file",
            )

            selected_treatment_path = os.path.join(
                upload_folder, selected_treatment_file
            )
            st.session_state["treatment"] = pd.read_csv(
                selected_treatment_path, header=None
            ).values


with treatment_col2:
    if hasTreatments:
        st.write("__Preview__")
        st.dataframe(st.session_state["treatment"], width=100, height=210)

st.divider()

hasCellLines = st.checkbox("Cell Lines")

cell_line_col1, cell_line_col2 = st.columns([0.8, 0.2])
with cell_line_col1:
    if hasCellLines:
        uploaded_labels = st.file_uploader(
            "Upload ___Cell Lines___ in one or multiple files (`csv`, `txt`)",
            type=["csv", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_labels:
            with st.spinner("Processing files..."):
                for uploaded_label in uploaded_labels:

                    handle_uploaded_file(uploaded_label, upload_folder)

                st.session_state["cell_lines"] = pd.read_csv(
                    uploaded_label, header=None
                ).values

        if st.session_state["config_option"] == "Choose an Uploaded File":

            # Display the files in a dropdown
            selected_cell_line_file = st.selectbox(
                "Choose a file:",
                get_csv_txt_files(upload_folder),
                key="selected_cell_line_file",
            )

            selected_cell_line_path = os.path.join(
                upload_folder, selected_cell_line_file
            )

            st.session_state["cell_lines"] = pd.read_csv(
                selected_cell_line_path, header=None
            ).values

with cell_line_col2:
    if hasCellLines:
        st.write("__Preview__")
        st.dataframe(st.session_state["cell_lines"], width=100, height=210)


# st.write(st.session_state["labels"])

if st.session_state.cells_list == True:
    st.warning("Please upload a zipped file of ROIs or a CSV/TXT file.")
    st.stop()


st.markdown("## Preview of Cell Data")

st.warning(
    "⚠️ This is a preview your uploaded raw data. We have not preprocessed your data yet to close the curves and remove duplicates. Please continue to the next page to preprocess your data."
)


# Sanity check visualization
cell_num = st.number_input(
    f"Visualize a cell. Pick a number between 0 and {len(st.session_state.cells_list)-1}",
    min_value=0,
)

# Sample data. Replace this with st.session_state.cells_list[cell_num]
x_data = st.session_state.cells_list[cell_num][:, 0]
y_data = st.session_state.cells_list[cell_num][:, 1]
z_data = [0] * len(
    x_data
)  # Using a constant value for z, placing the line on the "floor" of the 3D space

# Define a custom color for the line plot
line_color = "rgb(255, 0, 191)"  # Adjust the RGB values as per your preference

# Create a 3D trace for the cell data
trace = go.Scatter3d(
    x=x_data,
    y=y_data,
    z=z_data,
    mode="lines",
    line=dict(color=line_color),
)

# Create the layout for the 3D plot
layout = go.Layout(
    title="Preview of 2D Cell Data in 3D Space",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
)

# Create the Figure object and add the trace to it
fig = go.Figure(data=trace, layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


st.dataframe(st.session_state.cells_list[cell_num])


# config_option = st.radio(
#     "Select a Configuration File Option", ("Upload", "Choose a File")
# )

# if config_option == "Upload":

#     # Display the file uploader
#     uploaded_files = st.file_uploader(
#         "Upload a file",
#         type=["zip", "csv", "txt"],
#         accept_multiple_files=True,
#         key="file_uploader",
#     )

#     # if st.session_state["cells_list"] == True:
#     #     if not uploaded_files:
#     #         st.warning("Please upload a zipped file of ROIs or a CSV/TXT file.")
#     #         st.stop()
#     # Process the uploaded files
#     if uploaded_files is not None:
#         progress_bar = st.progress(0)
#         total_files = len(uploaded_files)
#         completed_files = 0

#         for uploaded_file in uploaded_files:
#             file_path = os.path.join(upload_folder, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             completed_files += 1
#             progress = int((completed_files / total_files) * 100)
#             progress_bar.progress(progress)
#         build_and_load_data(upload_folder)
#         previewVisualization(st.session_state["cells_list"])

# elif config_option == "Choose a File":
#     if upload_folder:
#         try:
#             # List all files in the directory
#             files = [f for f in os.listdir(upload_folder) if os.path.isfile(os.path.join(upload_folder, f))]

#             if not files:
#                 st.warning("No files found in the directory!")
#                 # st.stop()

#             # Display the files in a dropdown
#             selected_file = st.selectbox("Choose a file:", files)
#             st.session_state["selected_dataset"] = selected_file

#             st.info(f"You selected {selected_file}")

#             build_and_load_data(upload_folder)
#             previewVisualization(st.session_state["cells_list"])

#         except Exception as e:
#             st.error(f"An error occurred: {e}")


# Specify the folder path for file uploads and save run with date and time


# if uploaded_files is None:
#   st.warning('Please upload a zipped file of ROIs')
#   st.stop()


# st.write(f"File saved: {file_path}")


# fig, ax = plt.subplots()
# ax.plot(cells_list[cell_num][:, 0], cells_list[cell_num][:, 1])
# st.pyplot(fig)


# Define a custom color for the line plot
# line_color = "rgb(31, 119, 180)"  # Adjust the RGB values as per your preference

# # Create a trace for the cell data
# trace = go.Scatter(
#     x=cells_list[cell_num][:, 0],
#     y=cells_list[cell_num][:, 1],
#     mode="lines",
#     line=dict(color=line_color),
# )

# # Create the layout for the plot
# layout = go.Layout(
#     title="Cell Data",
#     xaxis=dict(title="X"),
#     yaxis=dict(title="Y"),
# )

# # Create the Figure object and add the trace to it
# fig = go.Figure(data=trace, layout=layout)

# # Display the Plotly figure using Streamlit
# st.plotly_chart(fig)
