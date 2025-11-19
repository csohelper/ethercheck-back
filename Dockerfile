# ===== Этап 1: базовый образ и установка зависимостей =====
FROM python:3.12-slim AS builder

# Установим рабочую директорию
WORKDIR /app

# Копируем только файл зависимостей сначала (для кэширования слоёв)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# ===== Этап 2: копирование исходного кода =====
FROM python:3.12-slim

# Рабочая директория в контейнере
WORKDIR /app

# Копируем установленные зависимости из builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем исходный проект
COPY . .

# Указываем порт, который будет слушать сервер
EXPOSE 8080

# Запуск приложения
CMD ["python", "main.py"]
