FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

COPY . /tmp/haca3

RUN RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENTRYPOINT ["python", "../code_ori/harmonize.py"]
