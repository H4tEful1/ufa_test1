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
Confusion Matrix:
[[1194  685  130]
 [ 536 3698  929]
 [ 165  955 3000]]

Classification Report:
              precision    recall  f1-score   support

           0     0.6301    0.5943    0.6117      2009
           1     0.6928    0.7163    0.7043      5163
           2     0.7391    0.7282    0.7336      4120

    accuracy                         0.6989     11292
   macro avg     0.6873    0.6796    0.6832     11292
weighted avg     0.6985    0.6989    0.6985     11292

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
