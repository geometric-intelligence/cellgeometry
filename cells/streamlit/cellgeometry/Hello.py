import streamlit as st




st.set_page_config(
    page_title="Welcome to Cell Shape Analysis",
    page_icon=":rocket:",
)


st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ![Asset 2](https://github.com/bioshape-lab/cells/assets/22850980/cf1ca7b1-b9d2-4055-b80d-e2bef7a2e796)
    
    
""")

import webbrowser

# Define the GitHub link
github_link = 'https://github.com/bioshape-lab/cells/tree/main/cells/streamlit'

# Create a button
if st.button('Go to GitHub'):
    # Open the GitHub link in a new browser tab
    webbrowser.open_new_tab(github_link)

st.markdown("""
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

The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code in both commercial and non-commercial projects. It provides you with the freedom to adapt the software to your needs, while also offering some protection against liability. It is one of the most commonly used licenses in the open-source community.
"""
)