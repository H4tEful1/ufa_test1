import json
from app.model import predict
from app.preprocessing import preprocess


json_data = '{"text": "замечательный fabric много details"}'

data = json.loads(json_data)

text = data["text"]

processed_text = preprocess(text)

print(f"Preprocessed Text: {processed_text}")

result = predict(processed_text)

print(result)
