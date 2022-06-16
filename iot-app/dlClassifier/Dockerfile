FROM python:3.8-buster

WORKDIR /app
ADD ./requirements.txt /app/requirements.txt

COPY requirements.txt .

RUN apt-get update \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install -r requirements.txt

ADD triplet_fp_UakNN_no_plot.py /app
ADD bin_files /app/bin_files
ADD data /app/data
ADD models /app/models
RUN dir

ENTRYPOINT [ "python3", "triplet_fp_UakNN_no_plot.py", "test"]