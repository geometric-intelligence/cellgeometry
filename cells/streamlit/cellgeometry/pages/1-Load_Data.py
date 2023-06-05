import os
import time
import sys
from utils.data_utils import (
    build_rois,
    find_all_instances,
    get_files_from_folder,
    check_file_extensions,
    parse_coordinates,
)
import plotly.graph_objects as go
import streamlit as st


st.sidebar.header("STEP 1: Load Data")
sys.path.append("/app/utils")


current_time = time.localtime()

year = time.strftime("%Y", current_time)
day_of_year = time.strftime("%j", current_time)
time_string = time.strftime("%H%M%S", current_time)

if "current_time_string" not in st.session_state:
    current_time_string = f"{year}{day_of_year}-{time_string}"
    st.session_state["current_time_string"] = current_time_string


if "cells_list" not in st.session_state:
    st.session_state["cells_list"] = True

st.write("# Load Your Cell Data 👋")

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


# # Global variable to store the file path
# upload_folder = None

# def handle_file_upload():
#     global upload_folder

#     # Specify the folder path for file uploads and save run with date and time
#     upload_folder = f"/app/data/run-{current_time_string}"

#     # Check if the upload folder exists, and create it if it doesn't
#     if not os.path.exists(upload_folder):
#         os.makedirs(upload_folder)
#         st.info(f"Upload folder created: {upload_folder}")

#     # Example: Print the file path
#     st.write("Uploaded File Path:", upload_folder)

# Specify the folder path for file uploads and save run with date and time
upload_folder = f"/app/data/run-{st.session_state.current_time_string}"

# Check if the upload folder exists, and create it if it doesn't
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)
    st.session_state["upload_folder"] = upload_folder
    st.info(f"Upload folder created: {upload_folder}")


# Display the file uploader
uploaded_files = st.file_uploader(
    "Upload a file",
    type=["zip", "csv", "txt"],
    accept_multiple_files=True,
    key="file_uploader",
)


# if uploaded_files is None:
#   st.warning('Please upload a zipped file of ROIs')
#   st.stop()

if st.session_state["cells_list"] == True:
    if not uploaded_files:
        st.warning("Please upload a zipped file of ROIs or a CSV/TXT file.")
        st.stop()


# Process the uploaded files
if uploaded_files is not None:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    completed_files = 0

    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        completed_files += 1
        progress = int((completed_files / total_files) * 100)
        progress_bar.progress(progress)
        # st.write(f"File saved: {file_path}")


# Get the list of files in the upload folder
files = get_files_from_folder(upload_folder)


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

st.markdown("## Preview of Cell Data")

st.warning(
    "⚠️ This is a preview your uploaded raw data. We have not preprocessed your data yet to close the curves and remove duplicates. Please continue to the next page to preprocess your data."
)

# Sanity check visualization
cell_num = st.number_input(
    f"Visualize a cell. Pick a number between 0 and {len(cells_list)-1}",
    min_value=0,
)

# fig, ax = plt.subplots()
# ax.plot(cells_list[cell_num][:, 0], cells_list[cell_num][:, 1])
# st.pyplot(fig)


# Define a custom color for the line plot
line_color = "rgb(31, 119, 180)"  # Adjust the RGB values as per your preference

# Create a trace for the cell data
trace = go.Scatter(
    x=cells_list[cell_num][:, 0],
    y=cells_list[cell_num][:, 1],
    mode="lines",
    line=dict(color=line_color),
)

# Create the layout for the plot
layout = go.Layout(
    title="Cell Data",
    xaxis=dict(title="X"),
    yaxis=dict(title="Y"),
)

# Create the Figure object and add the trace to it
fig = go.Figure(data=trace, layout=layout)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)
