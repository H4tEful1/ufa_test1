Цель проекта - анализ тональности (sentiment analysis) ru и eng текста (отрицательная/нейтральная/положительная)

Цель достигаяется используя предобученную модель distilbert, которая была дообучена на датасетах rusentitweet + tweet_eval(hugging face)

Решение включает:
- Предобработку текста (токенизация, нормализация, удаление стоп-слов)
- Модель классификации
- REST API на FastAPI
- Докеризированное приложение
- Набор тестов

На вход модель принимает .json структуры
{"text": "замечательный fabric так много details"}
Возвращает (пример)
{
  "sentiment": "positive/negative/neutral",
  "confidence": От 0 до 1
}

Полученные метрики на валидационной выборке:
![image](https://github.com/user-attachments/assets/0ef884af-aae2-4c54-af5a-3bc665445121)


Для работы данной модели необходимо:
- скачать ее отдельно по ссылке 
- видеокарта с compute_capability >= 30
  https://developer.nvidia.com/cuda-gpus

Инструкция по установке и запуску

Локальный запуск
Установите зависимости:

bash
pip install -r requirements.txt
Запустите сервер:

bash
uvicorn main:app --reload
API будет доступно по адресу: http://localhost:8000

Либо в директории проекта\app запустите dev_test.py

Запуск в Docker
Соберите образ:

bash (из директории проекта)
docker build -t sentiment-api .
Запустите контейнер:

bash
docker run -p 8000:8000 sentiment-api

Использование API
Отправьте POST-запрос на /predict с JSON-телом:

json
{
  "text": "Ваш текст для анализа"
}
Пример ответа:

json
{
  "sentiment": "positive",
  "confidence": 0.87
}

Документация API доступна по адресу http://localhost:8000/docs (Swagger UI)
