FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git

COPY . /tmp/haca3

RUN pip install /tmp/haca3 && rm -rf /tmp/haca3

ENTRYPOINT ["haca3-test"]
