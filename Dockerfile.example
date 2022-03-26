FROM python:3.7.5

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

#CMD ls
CMD ["python","main.py","-H","192.168.1.64","--definitions","./pg_config.yaml","--database-url","postgres://zalando:fya8ciOW87Bbunjhz43SqWgMGrESK8c7Dj0AqfiX5JuP33GIJDjlWCcOBvZCKc57@acid-minimal-cluster:5432/foo?sslmode=require"]
