FROM osnieltx/cuda-python3.8

USER root

COPY requirements.txt /
COPY script.py /

RUN python3.8 -m pip install -r requirements.txt

# CMD python3.9 script.py
