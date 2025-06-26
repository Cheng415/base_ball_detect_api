FROM python:3.10-slim

# 安裝系統相依套件
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
