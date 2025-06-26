# 使用 Python 3.10 基礎映像
FROM python:3.10-slim

# 安裝必要系統套件
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean

# 設定工作目錄
WORKDIR /app

# 複製需求檔與程式碼
COPY requirements.txt .
COPY . .

# 安裝 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 啟動 FastAPI 伺服器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
