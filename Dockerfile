FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu16.04

USER root

COPY requirements.txt /
COPY script.py /

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libssl-dev libjpeg-dev libreadline-dev libffi-dev wget \
    libbz2-dev libsqlite3-dev
RUN mkdir /python && cd /python
RUN wget https://www.python.org/ftp/python/3.8.19/Python-3.8.19.tgz
RUN tar -zxvf Python-3.8.19.tgz
RUN cd Python-3.8.19 && ls -lhR && ./configure --enable-optimizations && make install

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install --upgrade setuptools
RUN python3.8 -m pip install -r requirements.txt

# CMD python3.9 script.py
