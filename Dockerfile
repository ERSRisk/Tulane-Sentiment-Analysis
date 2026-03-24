FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/source ./source

ENV PYTHONPATH=/app

CMD ["python", "source/RSS_extraction.py"]
