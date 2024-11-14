FROM python:3.10.12

WORKDIR /var/apps

RUN mkdir -p /var/apps \
    && mkdir -p /var/apps/kg-rag

COPY . ./kg-rag

WORKDIR /var/apps/kg-rag

RUN pip install --no-cache-dir -r /var/apps/kg-rag/requirements.txt

CMD ["sleep", "infinity"]