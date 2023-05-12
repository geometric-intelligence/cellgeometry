import streamlit as st




st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to the Cell Shape Analysis App! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Geomstats is an open-source Python package for computations, statistics, and machine learning on nonlinear manifolds. Data from many application fields are elements of manifolds. For instance, the manifold of 3D rotations SO(3) naturally appears when performing statistical learning on articulated objects like the human spine or robotics arms. 
    **
    
    👈 Select a demo from the sidebar** 
"""
)