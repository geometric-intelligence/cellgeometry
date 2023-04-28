# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Run APT installs
RUN apt update && apt install -y git

# Set the working directory to /app
WORKDIR /app

# Copy the Poetry files into the container and install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy the entire project directory into the container
COPY cells .

# Expose the port that Streamlit listens on (8501 by default)
EXPOSE 8501

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app/utils"

# Set the entrypoint command to run the Streamlit app
CMD ["streamlit", "run", "Hello.py", "--server.port=8501"]

