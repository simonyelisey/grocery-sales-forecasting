# Use an official Python runtime as a parent image
FROM python:3.9.18-slim-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY ./configs ./configs
COPY ./data ./data
COPY ./grocery-sales-forecasting ./grocery-sales-forecasting
COPY ./models ./models
COPY ./sql ./sql
COPY ./main.py ./main.py
COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml
COPY ./requirements.txt ./requirements.txt

# Install any needed packages specified in pyproject.toml
RUN apt-get update && apt-get install -y \
    gcc python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install poetry
RUN poetry install
RUN pip install -r requirements.txt

# Run main.py when the container launches
CMD ["python3", "./main.py"]
