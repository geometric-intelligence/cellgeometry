import streamlit as st
import pandas as pd
from io import StringIO

st.write("# Welcome to the Cell Shear Analysis App! 👋")

st.markdown(
    """
    ## Data Source

> Ehsan Sadeghipour, Miguel A Garcia, William James Nelson, Beth L Pruitt (2018) Shear-induced damped oscillations in an epithelium depend on actomyosin contraction and E-cadherin cell adhesion eLife 7:e39640 https://doi.org/10.7554/eLife.39640

![](https://raw.githubusercontent.com/amilworks/ece594n/728845ba67ef604d307be98f78b872aa4d4052a4/hw_project/PredictingCellShear/figs/Graphical_Abstract_V1%404x.png)

# Introduction and Motivation

Cell-cell shear, or the action of cells sliding past each other, has roles in development, disease, and wound healing. Throughout development cells are moving past each other in every stage of development. These biomechanical cues have influences on differentiation, cell shape, behavior, the proteome, and the transcriptome. 

Previous research on shear focused on fluid shear so in this paper they focused on cell-cell shear which has been well characterized. Epithelial cells known as MDCK cells were used on a MEMS device which can be precisely displaced to create consistent cell-cell shear forces. Using new segmentation and machine learning techniques we are reanalyzing the data to use the changes in cell shape to predict cell behavior/migration.
   
"""
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)



st.sidebar.success("Select a demo above.")

