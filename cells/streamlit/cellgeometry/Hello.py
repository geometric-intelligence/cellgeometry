"""This module provides functionality for cell shape analysis using streamlit."""

import streamlit as st

# How we get the current user after successful login
from streamlit.web.server.websocket_headers import _get_websocket_headers

headers = _get_websocket_headers()
access_token = headers.get("X-Forwarded-User")
st.session_state["username"] = access_token

st.set_page_config(
    page_title="Welcome to Cell Shape Analysis",
    page_icon=":rocket:",
)


st.sidebar.title("Cell Shape Analysis")
st.sidebar.success("Select a demo above.")

st.success(f"Successfully logged in as __{access_token}__", icon="✅")

st.markdown(
    """
    ![Asset 5](https://github.com/bioshape-lab/cells/assets/22850980/344f448f-84a9-4f06-8527-8ddec210fb31)
    """
)

# https://geomstats.github.io/notebooks/10_practical_methods__shape_analysis.html
# https://arxiv.org/pdf/1803.10894.pdf
# https://arxiv.org/pdf/2209.09862.pdf

# Define the GitHub link
github_link = "https://github.com/bioshape-lab/cells/tree/main/cells/streamlit"

# Click this link to open the app in a new tab.
st.markdown(
    f'<a href="{github_link}" target="_blank">View on GitHub</a>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <meta property="og:title" content="CellGeometry">
    <meta property="og:description" content="A web-based app for Cell Shape Analysis">
    <meta property="og:image" content="https://raw.githubusercontent.com/bioshape-lab/cellgeometry/299d08de1aa3a1b3431ccd202f1e87530d2fd722/docs/static/img/cg-social-card-01.png">
    <meta property="og:url" content="https://cellgeometry.ece.ucsb.edu">
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ## 📝 Project Description

    This project focuses on the analysis and comparison of biological cell shapes using elastic metrics implemented in Geomstats. The shapes of biological cells are determined by various processes and biophysical forces, which play a crucial role in cellular functions. By utilizing quantitative measures that reflect cellular morphology, this project aims to leverage large-scale biological cell image data for morphological studies.

    The analysis of cell shapes has significant applications in various domains. One notable application is the accurate classification and discrimination of cancer cell lines treated with different drugs. Measures of irregularity and spreading of cells can provide valuable insights for understanding the effects of drug treatments.

    ## 🎯 Features

    - Quantitative analysis and comparison of biological cell shapes using Geomstats.
    - Utilization of elastic metrics implemented in Geomstats for shape analysis.
    - Calculation of measures reflecting cellular morphology, facilitating in-depth characterization of cell shapes.
    - Leveraging large-scale biological cell image data for comprehensive morphological studies.
    - Framework for optimal matching, deforming, and comparing cell shapes using geodesics and geodesic distances.
    - Visualization of cellular shape variations, aiding in the interpretation and communication of analysis results.
    - User-friendly Streamlit app interface for seamless analysis, visualization, and interaction with biological cell data.
    - Comprehensive set of tools and methods, empowering researchers and scientists in cellular biology to gain insights and make discoveries.

    ## 👈 Get Started

    Select a demo from the sidebar to start analyzing your cell data!

    ## 📄 License

    This project is licensed under the MIT License - see the LICENSE file for details.

    The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code in both commercial and non-commercial projects. It provides you with the freedom to adapt the software to your needs while also offering some protection against liability. It is one of the most commonly used licenses in the open-source community.
    """
)
