FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /tmp

COPY . /tmp

COPY requirements.txt /tmp

RUN pip install /tmp && rm -rf /tmp

EXPOSE 80

ENTRYPOINT ["python", "../haca3/harmonize.py"]
