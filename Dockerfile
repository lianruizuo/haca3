FROM python:3.10

WORKDIR /tmp

COPY . /tmp

COPY requirements.txt /tmp

RUN pip install /tmp && rm -rf /tmp

EXPOSE 80

ENTRYPOINT ["python", "../haca3/test.py"]
