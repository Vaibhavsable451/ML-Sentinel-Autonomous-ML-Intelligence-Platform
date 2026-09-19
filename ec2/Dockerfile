FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate dataset and artifacts if missing
RUN PYTHONPATH=. python data/generate_data.py && \
    PYTHONPATH=. python scripts/train_and_register.py

EXPOSE 8000 8501

CMD ["sh", "-c", "PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 & PYTHONPATH=. streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
