FROM python:3.9

#to prevent creating pyc files
ENV PYTHONDONTWRITEBYTECODE=1

#Setting PYTHONUNBUFFERED to a non empty value ensures that the python output is sent straight to terminal
#(e.g. your container log) without being first buffered and that you can see the output of your application 
#(e.g. django logs) in real time.
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/* && apt-get clean
WORKDIR /app 
COPY requirements.txt /app/
RUN pip install -r requirements.txt
#copy the current directory contents into docker
COPY . /app/

