.PHONY: stop build run all

# Stops the docker container
stop:
	-docker stop streamlit

# Builds the docker image
build:
	@docker build -t amilworks/streamlit-geomstats .

# Runs the docker container
run:
	@docker run -itp 8501:8501 --network pacmap2_default -d --rm --name streamlit -v ./cellgeometry:/app amilworks/streamlit-geomstats

# Calls all the above targets
all: stop build run

