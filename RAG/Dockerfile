FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-eng \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod +x start.sh

CMD ["./start.sh"]