# 使用 Python 3.10 基礎映像
FROM python:3.10-slim

# 安裝必要系統套件
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && apt-get clean

# 設定工作目錄
WORKDIR /app

# 複製需求檔與程式碼
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 開放 Cloud Run port
EXPOSE 8080

# 啟動 FastAPI 伺服器 (使用 PORT 環境變數)
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
