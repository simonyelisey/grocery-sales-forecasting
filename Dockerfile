# Use an official Python runtime as a parent image
FROM python:3.9.18-slim-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY ./configs ./configs
COPY ./data ./data
COPY ./grocery-sales-forecasting ./grocery-sales-forecasting
COPY ./models ./models
COPY ./main.py ./main.py
COPY ./requirements.txt ./requirements.txt
#COPY ./poetry.lock ./poetry.lock
#COPY ./pyproject.toml ./pyproject.toml

# Install any needed packages specified in pyproject.toml
#RUN pip install --upgrade pip
#RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update -y && sudo apt-get install gcc
RUN python3 -m pip install --no-cache-dir --upgrade pip \
      && python3 -m pip install -U setuptools \
      && python3 -m pip install --no-cache-dir -r requirements.txt
RUN dvc pull
#RUN pip install poetry
#RUN poetry install
#dvc pull

# Run app.py when the container launches
CMD ["python3", "./main.py"]
