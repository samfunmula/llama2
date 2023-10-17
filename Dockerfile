FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN apt update && \
    pip install --no-cache-dir -r requirements.txt

COPY src .

EXPOSE 8050

CMD python3 api.py
