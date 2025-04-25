FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt
#RUN python -m nltk.downloader punkt stopwords

CMD ["pytest"]

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
