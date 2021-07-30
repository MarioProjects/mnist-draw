FROM python:3.6-slim-buster

# Setting up Docker environment
WORKDIR /code

# Copy requirements file from current directory to file in
# containers code directory we have just created.
COPY requirements.txt requirements.txt

# Run and install all required modules in container
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y  # For OpenCV
RUN pip3 install -r requirements.txt

# Copy current directory files to containers code directory
COPY . .

# Export env variables.
ENV FLASK_RUN_PORT=8028

# RUN app.
CMD gunicorn --workers 2 --bind $(HOST):$FLASK_RUN_PORT app:app
