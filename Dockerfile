FROM python:3.9.21-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libgl1 \
    libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt --default-timeout=100 --retries=5 --no-cache-dir
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . /app
WORKDIR /app

CMD [ "python", "run.py"]