FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Code plus artifacts produced by `python -m src.train` (model + held-out replay data).
COPY src/ src/
COPY tests/ tests/
COPY app.py .
COPY models/ models/
COPY data/processed/ data/processed/

EXPOSE 8000 8501

# Default: API. docker-compose overrides the command for the dashboard.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
