# importing CUDA base image
FROM nvidia/cuda:12.5.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# place timezone data in /etc/timezone
ENV TZ=America

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install system dependencies
RUN apt-get update && \
    apt-get install -y sudo --no-install-recommends

RUN sudo apt-get update && \
    sudo apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        libnvidia-container1 \
        nvidia-container-runtime \
        libnvidia-container-tools --no-install-recommends

# creating the working directory
RUN mkdir -p /ser-wavelet

# setting the working directory
WORKDIR /ser-wavelet

# installing pip
RUN pip install --no-cache-dir -U pip

# copying all the files to the working directory
COPY . /ser-wavelet/

# installing requirements
RUN pip install -r requirements.txt