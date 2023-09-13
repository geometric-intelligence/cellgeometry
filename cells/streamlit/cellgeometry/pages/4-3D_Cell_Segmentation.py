import requests
import json
import streamlit as st
import numpy as np

# Define the API endpoint
url = "http://fastapi-app:8000/pacmap"

data = np.random.rand(590, 20)
data = data.tolist()
st.write(f"Data: {data}")
# Define the data you wish to send
data = {
    "data": data,
}

# Send a POST request to the API endpoint
response = requests.post(url, json=data)

# Check the response
if response.status_code == 200:
    reduced_data = response.json().get("reduced_data")
    st.write(f"Reduced Data: {reduced_data}")
else:
    st.write(f"Error {response.status_code}: {response.text}")


############################################################################################################
# Yes! you have the discrete_surfaces.py that is the generalization of discrete_curves.py
# https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/discrete_surfaces.py
# 1:12
# You can create the manifold of surfaces, and it also has an elastic metric.
# Basically, substituting:
# manifold = DiscreteCurves(…)
# by
# manifold = DiscreteSurfaces(…)
# Could be the only modification you need to make to your current code.
# Beware that the surface code is currently quite slow

# Here is a simple example of discrete surface (a cube) that you can load an play with:
# https://github.com/geomstats/geomstats/blob/2eeee177044e38080cc1004ae4a0bf8dd9ceb601/geomstats/datasets/utils.py#L452
############################################################################################################
