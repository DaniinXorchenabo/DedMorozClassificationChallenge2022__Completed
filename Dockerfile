FROM python:3.10.2-slim-buster
#RUN mkdir "app"
COPY requirements/prod.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
#WORKDIR /app
