FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y git
RUN apt update -y && apt install awscli -y
RUN pip3 install -r requirements.txt
CMD [ "python3", 'app.py' ]
