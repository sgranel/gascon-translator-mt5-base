FROM python:3.10-slim

WORKDIR /app

# Installer les dépendances système nécessaires à SentencePiece
RUN apt-get update && apt-get install -y \
    libsentencepiece-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier ton script
COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
