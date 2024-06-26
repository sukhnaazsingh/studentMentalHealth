FROM python:3.9

RUN apt-get -y update

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]