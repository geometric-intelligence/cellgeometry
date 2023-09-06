---
sidebar_position: 2
---

# Computing the Mean Cell Shape

This guide will walk you through each step in the process, ensuring you get the most out of our tool.

### 📜 Step-by-Step Walkthrough

1. **Initialization and Sidebar Header**:
   - The sidebar displays the header "STEP 2: Compute Mean Shape".

2. **Data Selection**:
   - The app first checks if you have uploaded or selected a dataset.
     - If not, a warning prompts you to upload or select data using the "Load Data" option.
     - If data is already uploaded, the filename is displayed for confirmation.

3. **Step Zero - Data Uploading**:
   - If you haven't uploaded your data, navigate to the "Load Data" page and follow the instructions to ensure proper formatting.

4. **Analyzing Cell Data**:
   - The app provides preprocessing steps, including interpolation, duplicate removal, and quotienting, to prepare your data for analysis.

5. **Sampling Points Selection**:
   - Use the slider to select the number of sampling points, which are crucial for the shape analysis. The default is set to 150.

6. **Data Preprocessing**:
   - Your data undergoes preprocessing to convert it into a suitable format for analysis.

7. **Exploring the Geodesic Trajectory** (Optional):
   - If you wish to explore the geodesic trajectory between two cell shapes, activate the "Explore Geodesic Trajectory" toggle in the sidebar.
   - Choose the desired treatment and cell line for your analysis.
   - Use the provided sliders to select cell indices.
   - The geodesic trajectory between the two chosen cell shapes is then visualized in a series of plots.

8. **Compute Mean Shape**:
   - Activate the "Compute Mean Shape" toggle in the sidebar to begin this analysis.
   - The computed mean shape will be displayed, which is essentially an average representation of all the cell shapes in your dataset.
   - Three plots provide visual insights:
     - A plot showcasing the mean estimate.
     - A combined plot illustrating individual cell shapes alongside the mean estimate.
     - A global mean shape superimposed on the entire dataset of cells.

### 🚀 Usage Tips:

- **Always Check Data**: Ensure you've uploaded the correct dataset before proceeding with the analysis.
- **Sampling Points**: Adjusting the number of sampling points can provide different levels of granularity in the analysis. However, higher values may increase computation time.
- **Geodesic Trajectory**: This visualization helps understand the transition between two cell shapes, useful for comparing different treatments or cell lines.

🔍 **Note**: Proper data preparation and understanding of each step are key to extracting meaningful insights from your cell shape data. Happy Analyzing! 🎉
