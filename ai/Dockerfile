FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY batch.py .
COPY prompt.txt metadata_prompt.txt ./

ENTRYPOINT ["python", "batch.py"]
CMD ["--watch"]
