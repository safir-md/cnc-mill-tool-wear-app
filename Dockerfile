FROM python:slim

WORKDIR /flask-app

ADD . /flask-app

RUN python3 -m pip --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD [ "python3", "wsgi.py" ]