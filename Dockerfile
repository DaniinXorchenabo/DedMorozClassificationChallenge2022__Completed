FROM python:3.10.2-slim-buster

COPY requirements/prod.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN mkdir "app"
WORKDIR /app
