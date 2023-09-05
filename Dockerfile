FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /tmp

COPY . /tmp

COPY requirements.txt /tmp

RUN pip install /tmp/haca3 && rm -rf /tmp/haca3

EXPOSE 80

ENTRYPOINT ["python", "../code_ori/harmonize.py"]
