FROM osnieltx/cuda-python3.8

USER root

COPY script.py /
COPY models.py /
COPY pyg.py /
