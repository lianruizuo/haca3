FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /tmp

COPY . /tmp

COPY requirements.txt /tmp

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENTRYPOINT ["python", "../code_ori/harmonize.py"]
