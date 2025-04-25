# Sentiment Analysis API 🌟

Проект для анализа тональности текста на русском и английском языках (negative/neutral/positive)

## 📌 О проекте

**Решение включает:**
- 🧹 Предобработку текста (токенизация, нормализация, очистка)
- 🤖 Модель классификации на базе дообученного DistilBERT
- 🚀 REST API на FastAPI
- 🐳 Docker-контейнеризацию
- ✅ Набор unit-тестов

**Технологии:**
- Python 3.10+
- PyTorch + Transformers
- FastAPI
- Docker

## 📊 Результаты модели

Модель дообучена на:
- [RusentiTweet](https://example.com)
- [TweetEval](https://huggingface.co/datasets/tweet_eval)

**Метрики на валидационной выборке:**

![Metrics](https://github.com/user-attachments/assets/0ef884af-aae2-4c54-af5a-3bc665445121)

## ⚙️ Требования

- Видеокарта с **compute_capability >= 3.0** ([проверить](https://developer.nvidia.com/cuda-gpus))
- CUDA 11.7+
- Модель (~250MB) - [скачать](https://drive.google.com/drive/folders/1-L73ZKS6f0RmWz3H0L7T5GLYXRhSvJRi)

## 🚀 Быстрый старт

### Локальная установка

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
uvicorn main:app --reload
Документация API доступна по адресу http://localhost:8000/docs (Swagger UI)

```
### Docker установка

```bash

# Сборка образа
docker build -t sentiment-api .

# Запуск контейнера
docker run -p 8000:8000 sentiment-api

📡 Использование API
Пример запроса:

json
POST /predict
{
  "text": "Отличный сервис, быстрая доставка!"
}
Пример ответа:

json
{
  "sentiment": "positive",
  "confidence": 0.92
}
```
📚 Документация

Интерактивная документация доступна после запуска:

Swagger UI: http://localhost:8000/docs

Redoc: http://localhost:8000/redoc

🧪 Тестирование из директории проекта
```bash
pytest 
```
