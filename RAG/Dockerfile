FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# 只裝執行期需要的套件；不裝編譯器（鏡像會小非常多）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-eng \
    libglib2.0-0 libsm6 libxext6 libxrender1 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先裝相依（可快取）
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 精準複製必要檔案（不要 COPY . .）
COPY bot.py .
COPY image.py .
COPY start.sh .
RUN chmod +x start.sh && mkdir -p image_dir

CMD ["./start.sh"]
