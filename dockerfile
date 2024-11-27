FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y build-essential

# рабочая дирректория
WORKDIR /app

# Установка PyTorch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Установка зависимостей
COPY requirements.txt .
RUN pip install -r requirements.txt

# Копируем проект
COPY . .

# Команда по умолчанию
CMD ["python", "main.py"]