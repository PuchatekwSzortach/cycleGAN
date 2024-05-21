# syntax = docker/dockerfile:experimental
FROM tensorflow/tensorflow:2.16.1-gpu

RUN apt update && apt install -y libgl1 libcairo2-dev vim

# Install python environment
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Setup bashrc
COPY ./docker/bashrc /root/.bashrc

# Setup PYTHONPATH
ENV PYTHONPATH=.

# Tensorflow keeps on using deprecated APIs ^^
ENV PYTHONWARNINGS="ignore::DeprecationWarning:tensorflow"

# # Set up working directory
WORKDIR /app
