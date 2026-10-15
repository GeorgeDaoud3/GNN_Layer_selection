FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
RUN apt-get update -y && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    build-essential  # Install dependencies for building extensions
RUN pip3 install torch==2.4.0
COPY *.txt /workspace/Code/
RUN pip3 install -r /workspace/Code/requirements.txt
RUN pip install jupyterlab notebook voila
RUN  apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata && apt-get install ffmpeg libsm6 libxext6  -y
#COPY *.txt /workspace/Code/