FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir cython numpy scipy scikit-learn
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN playwright install --with-deps

COPY pipeline/source ./source
COPY pipeline/resources ./pipeline/resources

ENV PYTHONPATH=/app

CMD ["python", "source/RSS_extraction.py"]
