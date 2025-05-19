FROM apache/airflow:2.7.3-python3.8

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt