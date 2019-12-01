FROM puckel/docker-airflow

USER root

RUN apt-get update
RUN apt-get update -yqq \
    && apt-get install -y gcc freetds-dev \
    && apt-get install -y git procps

RUN pip install --user psycopg2-binary
#install requirements only
RUN pip install -r requirements.txt


ENV AIRFLOW_HOME=/usr/local/airflow
COPY ./airflow.cfg /usr/local/airflow/airflow.cfg